"""Stage 8 (Person B, Task 1.3): Shapley values of the marketing channels.

Scope: ONLY the Shapley calculation from the coalition values of Task 1.2. No hybrid attribution,
first/last-touch or linear baselines, or budget allocation (later tasks).

Pipeline:  coalition_values.csv (V(S) for all 2^N coalitions) -> Shapley value per channel
           -> shapley_values.csv

Concepts
  Coalition S     A subset of the marketing channels (the players), stored as a frozenset using the
                  canonical representation of Task 1.1. Order and repetition are meaningless.
  V(S)            The coalition value of Task 1.2: the probability of EVENTUALLY reaching Conversion
                  from Start in Person A's Markov model when only the channels in S are kept and
                  every other channel is redirected to Null (the same policy as Person A's Removal
                  Effect). So V(S) is a conversion PROBABILITY, V(N) is the baseline conversion
                  probability, and V(empty set) is whatever conversion remains with no channel at all.
  Marginal        m_i(S) = V(S + {i}) - V(S): how much the conversion probability rises when channel
  contribution    i joins coalition S. It depends on who is already there (e.g. Social may add a lot
                  alone but little once Email is present), so no single marginal is "the" value of i.
  Shapley weight  w(s, n) = s! (n - s - 1)! / n!, where s = |S|. Imagine the n channels joining in a
                  uniformly random order. For channel i, the probability that exactly the channels of
                  S joined before it is s! (n-s-1)! / n!. The weights over all S not containing i
                  therefore sum to 1: they turn the marginals into a fair average over all orders.
  Shapley value   phi_i = sum over S subset of N\\{i} of w(|S|, n) * [V(S + {i}) - V(S)]
                  i.e. channel i's AVERAGE marginal contribution over all arrival orders.

Why Shapley values suit multi-touch attribution
  A conversion is produced by several touchpoints jointly, so credit cannot be read off one channel
  in isolation. Shapley is the unique division that is efficient (credit sums to the total), symmetric
  (identical channels get identical credit), gives nothing to a channel that never changes V (null
  player) and is additive across games. It accounts for every combination of channels, which a
  first/last-touch rule ignores.

Efficiency (checked in validation):  sum_i phi_i = V(N) - V(empty set).
  The Shapley values split exactly the conversion probability that the channels add over the
  no-channel scenario. (V(empty set) can be positive, e.g. direct Start -> Conversion; that part
  belongs to no channel.)

How to read the output
  phi_i is attribution contribution, measured in conversion-probability units, under the V(S)
  definition above. It is NOT historical revenue, spend or ROI: the dataset has no reliable
  channel-level revenue or spend and none is used or invented. Do not read phi as "revenue share".

Implementation notes
  * Values are looked up by frozenset, so S and S + {i} map to the right rows regardless of row
    order. A missing coalition raises ShapleyError; nothing is guessed or filled in.
  * Terms are enumerated in a fixed order (by subset size, then alphabetically) and added with
    math.fsum, so results are deterministic and accurately rounded.
  * Cost: n * 2^(n-1) terms, in line with the 2^N coalitions already required by Task 1.1/1.2.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial, fsum, isfinite
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from shapley_engine.coalition_generator import deserialize_coalition
from shapley_engine.coalition_value import DEFAULT_VALUES_PATH, load_coalition_values

DEFAULT_SHAPLEY_PATH = Path("data/processed/shapley_values.csv")
SHAPLEY_COLUMNS = ["channel", "shapley_value"]
EFFICIENCY_TOL = 1e-9
EMPTY = frozenset()


class ShapleyError(ValueError):
    """The coalition values are incomplete/invalid, or a Shapley result violates an invariant."""


# --------------------------------------------------------------------------- 1. value lookup
def _label(coalition: frozenset) -> str:
    return "{" + ", ".join(sorted(coalition)) + "}" if coalition else "{} (empty)"


def build_value_lookup(values: pd.DataFrame) -> dict:
    """Coalition-values table (Task 1.2 output) -> {frozenset of channels: V(S)}.

    Uses the 'coalition' (channels joined with '|', empty string = empty coalition) and 'value'
    columns, decoded with Task 1.1's `deserialize_coalition`. Rejects duplicates and non-finite or
    non-numeric values. Does NOT modify `values`.
    """
    for col in ("coalition", "value"):
        if col not in values.columns:
            raise ShapleyError(f"coalition values table has no {col!r} column")
    if not pd.api.types.is_numeric_dtype(values["value"]) or pd.api.types.is_bool_dtype(values["value"]):
        raise ShapleyError("the 'value' column must be numeric")
    lookup = {}
    for text, v in zip(values["coalition"], values["value"]):
        coalition = deserialize_coalition(text)
        if coalition in lookup:
            raise ShapleyError(f"duplicate coalition in values table: {_label(coalition)}")
        if not isfinite(float(v)):
            raise ShapleyError(f"V({_label(coalition)}) is not finite: {v!r}")
        lookup[coalition] = float(v)
    return lookup


def channels_of(lookup: Mapping) -> list:
    """The players: every channel that appears in any coalition, sorted alphabetically."""
    return sorted(set().union(*lookup)) if lookup else []


def _get(lookup: Mapping, coalition: frozenset) -> float:
    try:
        return lookup[coalition]
    except KeyError:
        raise ShapleyError(f"missing coalition value V({_label(coalition)}); "
                           "the table must contain all 2^N coalitions") from None


def check_complete(lookup: Mapping) -> None:
    """Require V(empty), V(N) and the full power set (exactly 2^N coalitions, every one present)."""
    channels = channels_of(lookup)
    n = len(channels)
    _get(lookup, EMPTY)
    _get(lookup, frozenset(channels))
    for k in range(n + 1):  # name the first missing coalition, which is more useful than a count
        for combo in combinations(channels, k):
            _get(lookup, frozenset(combo))
    if len(lookup) != 2 ** n:
        raise ShapleyError(f"expected 2^{n} = {2 ** n} coalition values, found {len(lookup)}")


# --------------------------------------------------------------------------- 2. the formula
def shapley_weight(n_players: int, coalition_size: int) -> float:
    """w = s! (n - s - 1)! / n!  for a coalition S of size s that does not contain the channel."""
    if n_players < 1 or not 0 <= coalition_size <= n_players - 1:
        raise ValueError(f"need n >= 1 and 0 <= s <= n-1, got n={n_players}, s={coalition_size}")
    return factorial(coalition_size) * factorial(n_players - coalition_size - 1) / factorial(n_players)


def shapley_value(lookup: Mapping, channel: str, channels: Iterable = None) -> float:
    """phi_channel = sum over S subset of N\\{channel} of w(|S|, n) * (V(S + {channel}) - V(S))."""
    players = sorted(set(channels)) if channels is not None else channels_of(lookup)
    if channel not in players:
        raise ShapleyError(f"{channel!r} is not a player")
    n = len(players)
    others = [p for p in players if p != channel]
    terms = []
    for size in range(n):  # |S| = 0 .. n-1
        w = shapley_weight(n, size)
        for combo in combinations(others, size):
            s = frozenset(combo)
            marginal = _get(lookup, s | {channel}) - _get(lookup, s)
            terms.append(w * marginal)
    return fsum(terms)


def compute_shapley_values(lookup: Mapping) -> dict:
    """Shapley value of every channel: {channel: phi}, channels in alphabetical order.

    Raises ShapleyError if any needed coalition value is missing. `lookup` is not modified.
    """
    check_complete(lookup)
    channels = channels_of(lookup)
    return {c: shapley_value(lookup, c, channels) for c in channels}


# --------------------------------------------------------------------------- 3. validation
def validate_shapley_values(shapley: Mapping, lookup: Mapping, tol: float = EFFICIENCY_TOL) -> dict:
    """Raise ShapleyError unless the Shapley values satisfy every invariant; return the checked numbers.

    Checks: V(empty) and V(N) exist; exactly one finite value per channel (no more, no fewer);
    efficiency sum(phi) == V(N) - V(empty) within `tol`.
    """
    channels = channels_of(lookup)
    v_empty = _get(lookup, EMPTY)
    v_grand = _get(lookup, frozenset(channels))
    if set(shapley) != set(channels) or len(shapley) != len(channels):
        raise ShapleyError(f"expected exactly one Shapley value for each of {channels}, got {sorted(shapley)}")
    for c, phi in shapley.items():
        if not isinstance(phi, (int, float)) or isinstance(phi, bool) or not isfinite(phi):
            raise ShapleyError(f"Shapley value of {c!r} is not a finite number: {phi!r}")
    total = fsum(shapley.values())
    expected = v_grand - v_empty
    if abs(total - expected) > tol:
        raise ShapleyError(f"efficiency violated: sum(phi) = {total!r} but V(N) - V(empty) = {expected!r}")
    return {"sum_shapley": total, "v_grand": v_grand, "v_empty": v_empty,
            "efficiency_gap": total - expected}


# --------------------------------------------------------------------------- 4. output
def shapley_to_frame(shapley: Mapping) -> pd.DataFrame:
    """{channel: phi} -> DataFrame[channel, shapley_value], sorted by channel (deterministic)."""
    items = sorted(shapley.items())
    return pd.DataFrame({"channel": [c for c, _ in items], "shapley_value": [float(p) for _, p in items]},
                        columns=SHAPLEY_COLUMNS)


def save_shapley_values(shapley: Mapping, path=DEFAULT_SHAPLEY_PATH) -> Path:
    """Write shapley_values.csv (creating the folder), full float precision, no rescaling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shapley_to_frame(shapley).to_csv(path, index=False)
    return path


def load_shapley_values(path=DEFAULT_SHAPLEY_PATH) -> dict:
    """Read shapley_values.csv back as {channel: phi}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/run_stage8.py` first.")
    df = pd.read_csv(path, dtype={"channel": str}, keep_default_na=False)
    if list(df.columns) != SHAPLEY_COLUMNS:
        raise ShapleyError(f"shapley file must have columns {SHAPLEY_COLUMNS}, got {list(df.columns)}")
    if df["channel"].duplicated().any():
        raise ShapleyError("shapley file lists a channel more than once")
    return dict(zip(df["channel"], df["shapley_value"].astype(float)))


# --------------------------------------------------------------------------- orchestration
def build_and_save(values_path=DEFAULT_VALUES_PATH, output_path=DEFAULT_SHAPLEY_PATH) -> dict:
    """coalition_values.csv -> Shapley values -> validate -> save -> re-read and validate again."""
    values = load_coalition_values(values_path)
    lookup = build_value_lookup(values)
    # Cross-check with Task 1.2's own record of Person A's baseline: V(N) must equal it.
    if "baseline_conversion_probability" in values.columns and len(values):
        grand = frozenset(channels_of(lookup))
        baseline = float(values["baseline_conversion_probability"].iloc[0])
        if abs(_get(lookup, grand) - baseline) > EFFICIENCY_TOL:
            raise ShapleyError(f"V(N) = {_get(lookup, grand)} differs from the baseline conversion "
                               f"probability {baseline} recorded in the values table")
    shapley = compute_shapley_values(lookup)
    checks = validate_shapley_values(shapley, lookup)
    out = save_shapley_values(shapley, output_path)
    validate_shapley_values(load_shapley_values(out), lookup)
    return {"shapley": shapley, "n_channels": len(shapley), "n_coalitions": len(lookup),
            "output_path": out, **checks}


def main() -> None:
    r = build_and_save()
    print(f"Number of channels (players):  {r['n_channels']}")
    print(f"Coalition values used:         {r['n_coalitions']} (= 2^N: {r['n_coalitions'] == 2 ** r['n_channels']})")
    print("Shapley values (conversion-probability units, NOT revenue):")
    for channel, phi in sorted(r["shapley"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {channel:<28} {phi:.10f}")
    print(f"Sum of Shapley values:         {r['sum_shapley']:.10f}")
    print(f"V(N) - V(empty):               {r['v_grand'] - r['v_empty']:.10f}   (V(N)={r['v_grand']:.10f}, V(empty)={r['v_empty']:.10f})")
    print(f"Efficiency gap:                {r['efficiency_gap']:.3e}")
    print("Validation: PASSED (efficiency, one finite value per channel; in memory and re-read from disk)")
    print(f"Output: {r['output_path']}")


if __name__ == "__main__":
    main()
