"""Task 1.4 - Shapley validation (Person B).

This module does NOT compute the production Shapley values.  It checks that the
Task 1.2 coalition values (V(S)) and the Task 1.3 Shapley values are
structurally and mathematically correct.

Checks performed (numbers are used in the printed report):

     1  coalition_completeness        2^N unique coalitions, none malformed/unexpected
     2  coalition_value_validity      numeric, finite, within [0, 1] (V is a probability)
     3  shapley_output_completeness   exactly one row per channel, no blank labels
     4  no_duplicate_channels
     5  no_missing_channels
     6  no_unexpected_channels
     7  shapley_values_numeric
     8  shapley_values_finite
     9  efficiency                    sum(phi) == V(N) - V(empty)
    10  independent_reference         phi recomputed by a separate method
    11  hand_calculated_examples      SYNTHETIC games with hand-derived answers
    12  symmetry                      symmetric channels -> equal phi; null channel -> phi = 0
    13  determinism                   same input (any row/channel order) -> same output

Negative Shapley values are NOT rejected: they are mathematically valid.
Sum(phi) is NOT required to equal 1; efficiency is sum(phi) = V(N) - V(empty).

Independence of the reference: the reference averages every channel's marginal
contribution over ALL orderings (permutations) of the channels (small N) or uses
the equivalent weight 1/(n * C(n-1, |S|)) (larger N).  It never imports or calls
the production Shapley code.

Expected input schema (inspected from the project description, not from the
files themselves - column names are auto-detected and can be overridden):

* coalition values : one column with the coalition (channels joined by one of
  ``| , ; +``; blank / ``∅`` / ``{}`` means the empty coalition) and one numeric
  value column.
* shapley values   : one channel column and one numeric Shapley column.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd

Coalition = FrozenSet[str]
GameValues = Dict[Coalition, float]
# Optional hook: the production Task 1.3 calculation wrapped as
# fn(channels, {coalition: value}) -> {channel: phi}.
ShapleyFn = Callable[[Sequence[str], Mapping[Coalition, float]], Mapping[str, float]]

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"

DEFAULT_TOL = 1e-9
MAX_CHANNELS = 20  # 2^20 coalitions is the largest game this validator handles
PERMUTATION_LIMIT = 8  # reference uses all N! orderings up to this many channels
SYMMETRY_LIMIT = 16  # pairwise symmetry scan is skipped above this many channels

RESERVED_STATES = {"start", "conversion", "null"}
EMPTY_TOKENS = {"", "∅", "empty", "emptyset", "empty_set", "none"}

COALITION_COLUMNS = ["coalition", "coalition_members", "members", "subset", "coalition_set", "channels"]
COALITION_VALUE_COLUMNS = [
    "value",
    "coalition_value",
    "v",
    "v_s",
    "vs",
    "conversion_probability",
    "probability",
]
SHAPLEY_CHANNEL_COLUMNS = ["channel", "channel_name", "player", "marketing_channel"]
SHAPLEY_VALUE_COLUMNS = ["shapley_value", "shapley", "phi", "shapley_raw", "value"]

CHECK_NAMES: Dict[int, str] = {
    1: "coalition_completeness",
    2: "coalition_value_validity",
    3: "shapley_output_completeness",
    4: "no_duplicate_channels",
    5: "no_missing_channels",
    6: "no_unexpected_channels",
    7: "shapley_values_numeric",
    8: "shapley_values_finite",
    9: "efficiency",
    10: "independent_reference",
    11: "hand_calculated_examples",
    12: "symmetry",
    13: "determinism",
}


class SchemaError(ValueError):
    """The input table does not have the columns the validator needs."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _close(a: float, b: float, tol: float) -> bool:
    """Tolerance-based float comparison (never exact equality)."""
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def format_coalition(coalition: Coalition) -> str:
    """Readable coalition, e.g. ``{A,B}``; the empty coalition is shown as ``∅``."""
    return "{" + ",".join(sorted(coalition)) + "}" if coalition else "∅"


def _examples(items: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def all_coalitions(channels: Sequence[str]) -> List[Coalition]:
    """The complete power set of ``channels`` in a deterministic order."""
    ordered = sorted(channels)
    return [frozenset(c) for r in range(len(ordered) + 1) for c in itertools.combinations(ordered, r)]


def parse_coalition_with_issue(cell: object) -> Tuple[Coalition, Optional[str]]:
    """Parse one coalition cell and report a problem if it is malformed.

    Accepts ``A|B``, ``A,B``, ``A;B``, ``A+B``, ``{A,B}``, ``[A, B]``; a missing
    cell, blank text, ``∅``, ``{}`` or ``empty`` is the empty coalition.
    Malformed: an empty member name (``A,,B``), a channel repeated inside one
    coalition, or a reserved state (Start / Conversion / Null) used as a channel.
    """
    if cell is None or (not isinstance(cell, (list, tuple, set, frozenset)) and pd.isna(cell)):
        return frozenset(), None
    if isinstance(cell, (list, tuple, set, frozenset)):
        parts = [str(x).strip() for x in cell]
    else:
        text = re.sub(r"[\[\]{}()\"']", "", str(cell)).strip()
        if text.lower() in EMPTY_TOKENS:
            return frozenset(), None
        parts = [p.strip() for p in re.split(r"[|,;+]", text)]
    issues: List[str] = []
    if any(p == "" for p in parts):
        issues.append("empty member name")
    names = [p for p in parts if p]
    if len(set(names)) != len(names):
        issues.append("channel repeated inside one coalition")
    reserved = sorted({p for p in names if p.lower() in RESERVED_STATES})
    if reserved:
        issues.append("reserved state used as channel: " + ",".join(reserved))
    members = frozenset(p for p in names if p.lower() not in RESERVED_STATES)
    return members, ("; ".join(issues) if issues else None)


def parse_coalition(cell: object) -> Coalition:
    """Parse one coalition cell into a frozenset of channel names."""
    return parse_coalition_with_issue(cell)[0]


def _resolve_column(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str], label: str) -> str:
    columns = list(df.columns)
    lowered = {str(c).strip().lower(): c for c in columns}
    if explicit is not None:
        if explicit in columns:
            return explicit
        if explicit.strip().lower() in lowered:
            return lowered[explicit.strip().lower()]
        raise SchemaError(f"Column '{explicit}' ({label}) not found. Columns present: {columns}")
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise SchemaError(
        f"Could not find the {label} column. Tried {list(candidates)}; columns present: {columns}. "
        "Pass the column name explicitly."
    )


def _numeric_column(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return (float array, mask of cells that are present but not numeric)."""
    # Python's float() is exactly rounded (pandas' string parser can be 1 ulp off) and it
    # understands "nan" / "inf", so those are reported as non-finite rather than non-numeric.
    values: List[float] = []
    garbage: List[bool] = []
    for cell in series.astype(object):
        if cell is None or (isinstance(cell, str) and not cell.strip()) or (not isinstance(cell, str) and pd.isna(cell)):
            values.append(float("nan"))
            garbage.append(False)
            continue
        try:
            values.append(float(cell.strip() if isinstance(cell, str) else cell))
            garbage.append(False)
        except (TypeError, ValueError):
            values.append(float("nan"))
            garbage.append(True)
    return np.array(values, dtype=float), np.array(garbage, dtype=bool)


# --------------------------------------------------------------------------- #
# Independent reference Shapley calculation
# --------------------------------------------------------------------------- #
def _require_complete(players: Sequence[str], values: Mapping[Coalition, float]) -> None:
    missing = [format_coalition(s) for s in all_coalitions(players) if s not in values]
    if missing:
        raise ValueError(f"Game is missing {len(missing)} coalition(s): {_examples(missing)}")


def reference_shapley_by_permutations(players: Sequence[str], values: Mapping[Coalition, float]) -> Dict[str, float]:
    """Shapley value = average marginal contribution over all N! channel orderings."""
    players = list(players)
    if not players:
        raise ValueError("Cannot compute Shapley values for a game with no channels")
    _require_complete(players, values)
    totals = {p: 0.0 for p in players}
    for order in itertools.permutations(players):
        coalition: Coalition = frozenset()
        previous = values[coalition]
        for player in order:
            coalition = coalition | {player}
            current = values[coalition]
            totals[player] += current - previous
            previous = current
    n_orders = math.factorial(len(players))
    return {p: totals[p] / n_orders for p in players}


def reference_shapley_by_subsets(players: Sequence[str], values: Mapping[Coalition, float]) -> Dict[str, float]:
    """Equivalent formula with weight 1 / (n * C(n-1, |S|)) (== |S|!(n-|S|-1)!/n!)."""
    players = list(players)
    n = len(players)
    if n == 0:
        raise ValueError("Cannot compute Shapley values for a game with no channels")
    _require_complete(players, values)
    result: Dict[str, float] = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for size in range(len(others) + 1):
            weight = 1.0 / (n * math.comb(n - 1, size))
            for combo in itertools.combinations(others, size):
                s = frozenset(combo)
                total += weight * (values[s | {i}] - values[s])
        result[i] = total
    return result


def reference_shapley(players: Sequence[str], values: Mapping[Coalition, float]) -> Dict[str, float]:
    """Reference Shapley values (permutation average for small games, subset form otherwise)."""
    if 0 < len(players) <= PERMUTATION_LIMIT:
        return reference_shapley_by_permutations(players, values)
    return reference_shapley_by_subsets(players, values)


# --------------------------------------------------------------------------- #
# Hand-calculated SYNTHETIC games
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HandGame:
    """A small SYNTHETIC game whose Shapley values were derived by hand."""

    name: str
    players: Tuple[str, ...]
    values: Dict[Coalition, float]
    expected: Dict[str, float]


def _make_values(players: Sequence[str], rule: Callable[[Coalition], float]) -> GameValues:
    return {s: float(rule(s)) for s in all_coalitions(players)}


def _hand_games() -> List[HandGame]:
    f = frozenset
    two_player = {f(): 0.0, f("A"): 10.0, f("B"): 20.0, f("AB"): 40.0}
    return [
        # phi_A = 1/2*(0.4-0) + 1/2*(0.8-0.2) = 0.5 ; phi_B = 1/2*(0.2-0) + 1/2*(0.8-0.4) = 0.3
        HandGame(
            "two_channel_project_example",
            ("A", "B"),
            {f(): 0.0, f("A"): 0.4, f("B"): 0.2, f("AB"): 0.8},
            {"A": 0.5, "B": 0.3},
        ),
        # phi_A = 1/2*10 + 1/2*(40-20) = 15 ; phi_B = 1/2*20 + 1/2*(40-10) = 25
        HandGame("two_channel_synergy", ("A", "B"), dict(two_player), {"A": 15.0, "B": 25.0}),
        # phi_A = 1/2*10 + 1/2*(6+2) = 9 ; phi_B = 1/2*(-2) + 1/2*(6-10) = -3  (negative phi)
        HandGame(
            "two_channel_negative_phi",
            ("A", "B"),
            {f(): 0.0, f("A"): 10.0, f("B"): -2.0, f("AB"): 6.0},
            {"A": 9.0, "B": -3.0},
        ),
        # probability-valued game with a negative phi: phi_A=0.3+0.15=0.45, phi_B=0.05-0.10=-0.05
        HandGame(
            "probability_game_negative_phi",
            ("A", "B"),
            {f(): 0.0, f("A"): 0.6, f("B"): 0.1, f("AB"): 0.4},
            {"A": 0.45, "B": -0.05},
        ),
        # V(empty) > 0 (direct Start->Conversion): phi_A=1/2*0.3+1/2*0.5=0.4, phi_B=1/2*0.2+1/2*0.4=0.3
        HandGame(
            "positive_empty_coalition",
            ("A", "B"),
            {f(): 0.2, f("A"): 0.5, f("B"): 0.4, f("AB"): 0.9},
            {"A": 0.4, "B": 0.3},
        ),
        # majority game: all three symmetric -> 1/3 each
        HandGame(
            "three_channel_majority_symmetric",
            ("A", "B", "C"),
            _make_values(("A", "B", "C"), lambda s: 1.0 if len(s) >= 2 else 0.0),
            {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        ),
        # glove game: R is needed plus one L.  phi_R = 4/6 = 2/3, phi_L1 = phi_L2 = 1/6
        HandGame(
            "three_channel_glove",
            ("L1", "L2", "R"),
            _make_values(("L1", "L2", "R"), lambda s: 1.0 if ("R" in s and ("L1" in s or "L2" in s)) else 0.0),
            {"L1": 1 / 6, "L2": 1 / 6, "R": 2 / 3},
        ),
        # channel C adds nothing (null player): phi_C = 0, others as in the synergy game
        HandGame(
            "three_channel_null_player",
            ("A", "B", "C"),
            _make_values(("A", "B", "C"), lambda s: two_player[f(s - {"C"})]),
            {"A": 15.0, "B": 25.0, "C": 0.0},
        ),
    ]


HAND_GAMES: List[HandGame] = _hand_games()


def check_hand_examples(shapley_fn: Optional[ShapleyFn] = None, tol: float = DEFAULT_TOL) -> Tuple[bool, str]:
    """Check the reference (and optionally the production function) on SYNTHETIC hand games."""
    problems: List[str] = []
    for game in HAND_GAMES:
        reference = reference_shapley(game.players, game.values)
        for player, expected in game.expected.items():
            if not _close(reference[player], expected, 1e-12):
                problems.append(f"{game.name}: reference phi_{player}={reference[player]:.12g}, hand value {expected:.12g}")
        if shapley_fn is not None:
            try:
                produced = dict(shapley_fn(list(game.players), dict(game.values)))
            except Exception as exc:  # report, do not crash the validator
                problems.append(f"{game.name}: Task 1.3 function raised {type(exc).__name__}: {exc}")
                continue
            for player, expected in game.expected.items():
                if player not in produced or not _close(float(produced[player]), expected, tol):
                    problems.append(f"{game.name}: Task 1.3 phi_{player}={produced.get(player)}, hand value {expected:.12g}")
    scope = "reference and Task 1.3 function" if shapley_fn is not None else "reference only; no Task 1.3 function supplied"
    if problems:
        return False, f"SYNTHETIC games, {scope}: " + _examples(problems, 4)
    return True, f"{len(HAND_GAMES)} SYNTHETIC hand-calculated games match ({scope})"


def check_symmetry_and_null_players(
    channels: Sequence[str], values: Mapping[Coalition, float], phi: Mapping[str, float], tol: float
) -> Tuple[bool, str]:
    """Symmetric channels must have equal phi; a channel adding nothing must have phi = 0."""
    channels = sorted(channels)
    if len(channels) > SYMMETRY_LIMIT:
        return True, f"not scanned: N={len(channels)} exceeds {SYMMETRY_LIMIT}"
    problems: List[str] = []
    n_pairs = 0
    for i, j in itertools.combinations(channels, 2):
        rest = [c for c in channels if c not in (i, j)]
        symmetric = all(
            _close(values[frozenset(s) | {i}], values[frozenset(s) | {j}], tol)
            for size in range(len(rest) + 1)
            for s in itertools.combinations(rest, size)
        )
        if symmetric:
            n_pairs += 1
            if not _close(phi[i], phi[j], tol):
                problems.append(f"{i} and {j} are symmetric but phi differ ({phi[i]:.10g} vs {phi[j]:.10g})")
    n_null = 0
    for i in channels:
        rest = [c for c in channels if c != i]
        null_player = all(
            _close(values[frozenset(s) | {i}], values[frozenset(s)], tol)
            for size in range(len(rest) + 1)
            for s in itertools.combinations(rest, size)
        )
        if null_player:
            n_null += 1
            if not _close(phi[i], 0.0, tol):
                problems.append(f"{i} adds nothing to any coalition but phi={phi[i]:.10g}")
    if problems:
        return False, _examples(problems, 4)
    return True, f"{n_pairs} symmetric pair(s) and {n_null} null channel(s) found in the data, all consistent"


# --------------------------------------------------------------------------- #
# Report objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CheckResult:
    number: int
    name: str
    status: str
    detail: str


@dataclass
class ValidationReport:
    results: List[CheckResult]
    channels: List[str]
    n_coalitions: int
    columns_used: Dict[str, str]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.status == PASS for r in self.results)

    def counts(self) -> Dict[str, int]:
        counter = Counter(r.status for r in self.results)
        return {PASS: counter[PASS], FAIL: counter[FAIL], SKIPPED: counter[SKIPPED]}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"check_number": r.number, "check": r.name, "status": r.status, "detail": r.detail} for r in self.results]
        )

    def format(self) -> str:
        lines = [
            f"Channels (N={len(self.channels)}): {', '.join(self.channels) if self.channels else '-'}",
            f"Coalition rows read: {self.n_coalitions}",
            "Columns used: " + ", ".join(f"{k}='{v}'" for k, v in self.columns_used.items()),
            "",
        ]
        for r in self.results:
            lines.append(f"[{r.status:<7}] {r.number:>2}  {r.name}: {r.detail}")
        c = self.counts()
        lines.append("")
        lines.append(f"Summary: {c[PASS]} passed, {c[FAIL]} failed, {c[SKIPPED]} skipped")
        lines.append("OVERALL: " + ("VALIDATION PASSED" if self.passed else "VALIDATION FAILED"))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main validation
# --------------------------------------------------------------------------- #
def validate_shapley_outputs(
    coalition_df: pd.DataFrame,
    shapley_df: pd.DataFrame,
    *,
    expected_channels: Optional[Sequence[str]] = None,
    baseline: Optional[float] = None,
    baseline_tol: float = 1e-6,
    tol: float = DEFAULT_TOL,
    coalition_col: Optional[str] = None,
    value_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    shapley_col: Optional[str] = None,
    shapley_fn: Optional[ShapleyFn] = None,
) -> ValidationReport:
    """Run all 13 checks on a coalition-value table and a Shapley table.

    ``expected_channels`` - the channel set the game should have; by default the
    channels that appear in the coalition table.
    ``baseline`` - Person A's baseline conversion probability; if given, V(N) must
    match it within ``baseline_tol``.
    ``shapley_fn`` - optional Task 1.3 calculation (see ``ShapleyFn``); if given it is
    also run on the SYNTHETIC hand games (checks 11 and 13).
    """
    ccol = _resolve_column(coalition_df, coalition_col, COALITION_COLUMNS, "coalition")
    vcol = _resolve_column(coalition_df, value_col, COALITION_VALUE_COLUMNS, "coalition value")
    chcol = _resolve_column(shapley_df, channel_col, SHAPLEY_CHANNEL_COLUMNS, "channel")
    pcol = _resolve_column(shapley_df, shapley_col, SHAPLEY_VALUE_COLUMNS, "Shapley value")

    results: List[CheckResult] = []

    def add(number: int, ok: Optional[bool], detail: str) -> bool:
        status = SKIPPED if ok is None else (PASS if ok else FAIL)
        results.append(CheckResult(number, CHECK_NAMES[number], status, detail))
        return ok is True

    # ---- parse coalition table -------------------------------------------------
    parsed = [parse_coalition_with_issue(c) for c in coalition_df[ccol]]
    members: List[Coalition] = [p[0] for p in parsed]
    malformed = [f"row {idx}: '{coalition_df[ccol].iloc[idx]}' ({issue})" for idx, (_, issue) in enumerate(parsed) if issue]
    derived = sorted(set().union(*members)) if members else []
    channels = sorted({str(c).strip() for c in expected_channels}) if expected_channels is not None else derived
    n = len(channels)
    channel_set = set(channels)

    # ---- 1. coalition completeness --------------------------------------------
    complete = False
    if n == 0:
        add(1, False, "no marketing channels found in the coalition table")
    elif n > MAX_CHANNELS:
        add(1, False, f"N={n} exceeds the supported maximum of {MAX_CHANNELS} channels")
    else:
        required = set(all_coalitions(channels))
        counts = Counter(members)
        duplicates = sorted(format_coalition(k) for k, v in counts.items() if v > 1)
        missing = sorted(format_coalition(k) for k in required - set(members))
        extra = sorted({m for k in members for m in k} - channel_set)
        problems: List[str] = []
        if len(members) != 2**n:
            problems.append(f"{len(members)} rows, expected 2^{n} = {2**n}")
        if missing:
            problems.append(f"{len(missing)} missing coalition(s): {_examples(missing)}")
        if duplicates:
            problems.append(f"{len(duplicates)} duplicate coalition(s): {_examples(duplicates)}")
        if extra:
            problems.append(f"unexpected channel(s) inside coalitions: {_examples(extra)}")
        if malformed:
            problems.append(f"{len(malformed)} malformed coalition cell(s): {_examples(malformed, 3)}")
        complete = add(
            1,
            not problems,
            "; ".join(problems) if problems else f"N={n}; {len(members)} coalitions = 2^{n}, none missing, duplicated or malformed",
        )

    # ---- 2. coalition value validity -------------------------------------------
    v_arr, v_garbage = _numeric_column(coalition_df[vcol])
    n_garbage = int(v_garbage.sum())
    n_nonfinite = int((~np.isfinite(v_arr) & ~v_garbage).sum())
    finite_vals = v_arr[np.isfinite(v_arr)]
    n_range = int(((finite_vals < -tol) | (finite_vals > 1 + tol)).sum())
    problems = []
    if n_garbage:
        problems.append(f"{n_garbage} non-numeric value(s)")
    if n_nonfinite:
        problems.append(f"{n_nonfinite} NaN/missing/infinite value(s)")
    if n_range:
        problems.append(f"{n_range} value(s) outside [0, 1] (V is a conversion probability)")
    values: Optional[GameValues] = None
    if complete and not problems:
        values = {k: float(v) for k, v in zip(members, v_arr)}
    note = ""
    if baseline is not None:
        grand = frozenset(channels)
        if values is not None:
            v_grand = values[grand]
            if not _close(v_grand, baseline, baseline_tol):
                problems.append(f"V(N)={v_grand:.10g} differs from baseline {baseline:.10g} (tol {baseline_tol})")
            else:
                note = f"; V(N)={v_grand:.10g} matches baseline {baseline:.10g} (tol {baseline_tol})"
        else:
            problems.append("baseline comparison not possible (coalition table invalid)")
    elif values is not None:
        note = f"; V(empty)={values[frozenset()]:.10g}, V(N)={values[frozenset(channels)]:.10g}; baseline not supplied"
    add(
        2,
        not problems,
        "; ".join(problems) if problems else f"{len(v_arr)} values numeric, finite, within [0,1]" + note,
    )
    if not problems and not complete:
        values = None

    # ---- 3-8. Shapley table ------------------------------------------------------
    labels = [None if pd.isna(x) else str(x).strip() for x in shapley_df[chcol]]
    named = [lab for lab in labels if lab]
    n_blank = len(labels) - len(named)
    p_arr, p_garbage = _numeric_column(shapley_df[pcol])

    problems = []
    if len(labels) != n:
        problems.append(f"{len(labels)} row(s), expected {n} (one per channel)")
    if n_blank:
        problems.append(f"{n_blank} row(s) with a blank channel name")
    ok3 = add(3, not problems, "; ".join(problems) if problems else f"{len(labels)} rows, one per channel")

    dup = sorted(k for k, v in Counter(named).items() if v > 1)
    ok4 = add(4, not dup, f"duplicate channel(s): {_examples(dup)}" if dup else "every channel appears once")

    missing_ch = sorted(channel_set - set(named))
    ok5 = add(5, not missing_ch, f"missing channel(s): {_examples(missing_ch)}" if missing_ch else "no channel is missing")

    unexpected = sorted(set(named) - channel_set)
    ok6 = add(6, not unexpected, f"unexpected channel(s): {_examples(unexpected)}" if unexpected else "no unexpected channel")

    n_bad_num = int(p_garbage.sum())
    ok7 = add(7, n_bad_num == 0, f"{n_bad_num} non-numeric Shapley value(s)" if n_bad_num else "all Shapley values numeric")

    n_bad_fin = int((~np.isfinite(p_arr) & ~p_garbage).sum())
    ok8 = add(
        8,
        n_bad_fin == 0,
        f"{n_bad_fin} NaN/missing/infinite Shapley value(s)" if n_bad_fin else "all Shapley values finite (negative values are allowed)",
    )

    phi: Optional[Dict[str, float]] = None
    if all([ok3, ok4, ok5, ok6, ok7, ok8]):
        phi = {lab: float(v) for lab, v in zip(labels, p_arr)}

    need_game = "needs a valid coalition table (checks 1-2)"
    need_phi = "needs a valid Shapley table (checks 3-8)"

    def prerequisite() -> Optional[str]:
        if values is None and phi is None:
            return f"{need_game} and {need_phi}"
        if values is None:
            return need_game
        if phi is None:
            return need_phi
        return None

    # ---- 9. efficiency -----------------------------------------------------------
    reason = prerequisite()
    if reason:
        add(9, None, "skipped: " + reason)
    else:
        grand_gain = values[frozenset(channels)] - values[frozenset()]
        total = math.fsum(phi.values())
        add(
            9,
            _close(total, grand_gain, tol),
            f"sum(phi)={total:.12g} vs V(N)-V(empty)={grand_gain:.12g}, |diff|={abs(total - grand_gain):.3e}, tol={tol:g} "
            "(sum(phi) need not equal 1)",
        )

    # ---- 10. independent reference ----------------------------------------------
    if reason:
        add(10, None, "skipped: " + reason)
    else:
        reference = reference_shapley(channels, values)
        bad = [c for c in channels if not _close(phi[c], reference[c], tol)]
        max_diff = max(abs(phi[c] - reference[c]) for c in channels)
        method = "all-orderings average" if n <= PERMUTATION_LIMIT else "subset formula 1/(n*C(n-1,|S|))"
        if bad:
            detail = "; ".join(f"{c}: file {phi[c]:.10g} vs reference {reference[c]:.10g}" for c in bad[:4])
        else:
            detail = f"all {n} channels match the independent reference ({method}); max |diff|={max_diff:.3e}"
        add(10, not bad, detail)

    # ---- 11. hand-calculated examples (SYNTHETIC) --------------------------------
    ok, detail = check_hand_examples(shapley_fn, tol)
    add(11, ok, detail)

    # ---- 12. symmetry / null players ----------------------------------------------
    if reason:
        add(12, None, "skipped: " + reason)
    else:
        ok, detail = check_symmetry_and_null_players(channels, values, phi, tol)
        add(12, ok, detail)

    # ---- 13. determinism -----------------------------------------------------------
    parts: List[str] = []
    problems = []
    if values is not None:
        first = reference_shapley(channels, values)
        again = reference_shapley(channels, values)
        if first != again:
            problems.append("reference changed between identical runs")
        rows = list(values.items())
        random.Random(0).shuffle(rows)
        shuffled = reference_shapley(list(reversed(channels)), dict(rows))
        if not all(_close(first[c], shuffled[c], 1e-12) for c in channels):
            problems.append("reference depends on row/channel order")
        parts.append("reference identical on repeat and under row/channel reordering")
    if shapley_fn is not None:
        game = HAND_GAMES[0]
        try:
            run1 = dict(shapley_fn(list(game.players), dict(game.values)))
            run2 = dict(shapley_fn(list(game.players), dict(game.values)))
        except Exception as exc:
            problems.append(f"Task 1.3 function raised {type(exc).__name__}: {exc}")
        else:
            if run1 != run2:
                problems.append("Task 1.3 function gave different results on identical input")
            else:
                parts.append("Task 1.3 function identical on repeat (SYNTHETIC game)")
    if problems:
        add(13, False, "; ".join(problems))
    elif parts:
        add(13, True, "; ".join(parts))
    else:
        add(13, None, "skipped: " + (need_game if values is None else "nothing to run"))

    results.sort(key=lambda r: r.number)
    return ValidationReport(
        results=results,
        channels=channels,
        n_coalitions=len(members),
        columns_used={"coalition": ccol, "value": vcol, "channel": chcol, "shapley": pcol},
    )


def _read_csv(path: Path, description: str, producer_hint: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found ({description}). {producer_hint}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_shapley_files(
    coalition_values_path: Path,
    shapley_values_path: Path,
    **kwargs: object,
) -> ValidationReport:
    """Load the two project CSVs and run :func:`validate_shapley_outputs`."""
    coalition_df = _read_csv(
        coalition_values_path,
        "Task 1.2 coalition values",
        "Run the Task 1.2 stage that produces it first (e.g. python scripts/run_stage7.py).",
    )
    shapley_df = _read_csv(
        shapley_values_path,
        "Task 1.3 Shapley values",
        "Run the Task 1.3 stage that produces it first (e.g. python scripts/run_stage8.py).",
    )
    return validate_shapley_outputs(coalition_df, shapley_df, **kwargs)
