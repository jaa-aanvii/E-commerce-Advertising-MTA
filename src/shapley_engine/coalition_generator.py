"""Stage 6 (Person B, Task 1.1): players and coalitions for the Shapley / game-theory part.

Scope: ONLY "who are the players" and "what are all the possible coalitions".
NO coalition value V(S), marginal contributions, Shapley values, hybrid attribution or budget
allocation (later tasks). This module reads Person A's journeys.csv and never modifies it.

Pipeline:  journeys.csv -> unique marketing channels (players) -> power set -> coalitions.csv

Vocabulary
  Player     A marketing channel that actually appears in the journeys.
  Coalition  Any subset S of the players. Order and repetition are meaningless, so it is a set.
  Power set  The set of ALL subsets of the players, from the empty coalition to the grand coalition.
             N players -> 2^N coalitions, because each player is independently in or out of a subset.

Why Start, Conversion and Null are NOT players
  They are structural states of the journey model (Person A's Markov chain): Start is where every
  journey begins, Conversion and Null are the two absorbing endings. They are not something a
  marketing budget can be spent on, so they cannot "cooperate" to produce a conversion. Only the
  channels between Start and the terminal state are players.

Representation
  In memory: frozenset[str]. {"Google", "Email"} and {"Email", "Google"} are the same coalition,
  and frozensets are hashable (usable as dict keys when V(S) is added later).
  On disk (CSV): a separate string produced by `serialize_coalition`: channel names sorted
  alphabetically and joined with "|", e.g. "Email|Google". The EMPTY coalition is written as an
  EMPTY STRING. Reading it back with pandas needs keep_default_na=False (see `load_coalitions`),
  otherwise the empty field becomes NaN.

Ordering (deterministic, never depends on Python's set ordering)
  Coalitions are ordered by (size, sorted channel names): the empty set first, then singletons
  alphabetically, then pairs in lexicographic order, ..., the grand coalition last.

Scalability
  The power set is O(2^N) in both time and memory. `DEFAULT_MAX_CHANNELS` (20, about one million
  coalitions) is a safety limit against accidents, not a claim that larger N is practical. It is
  configurable per call and nothing is ever silently truncated.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mta.journey_builder import CONVERSION, DEFAULT_OUTPUT_PATH, NULL, PATH_SEP, START
from mta.markov_engine import load_journeys, parse_path

STRUCTURAL_STATES = (START, CONVERSION, NULL)
COALITION_SEP = "|"
DEFAULT_COALITIONS_PATH = Path("data/processed/coalitions.csv")
COALITION_COLUMNS = ["coalition_id", "coalition_size", "coalition"]
DEFAULT_MAX_CHANNELS = 20  # 2^20 = 1,048,576 coalitions; raise explicitly if you really need more

Coalition = frozenset  # frozenset[str]


class CoalitionValidationError(ValueError):
    """A generated (or saved) coalition collection violates a Task 1.1 invariant."""


class CoalitionLimitError(ValueError):
    """The number of channels would produce an impractically large power set."""


# --------------------------------------------------------------------------- 1. channels
def _states_of(item) -> list:
    """One journey-like item -> list of state names.

    Accepts a single state name ("Google"), a path string ("Start > Google > Conversion"),
    or a sequence of state names (["Start", "Google", "Conversion"]).
    """
    if isinstance(item, str):
        return item.split(PATH_SEP) if PATH_SEP in item else [item]
    if isinstance(item, Iterable):
        states = list(item)
        for s in states:
            if not isinstance(s, str):
                raise TypeError(f"state names must be strings, got {s!r} ({type(s).__name__})")
        return states
    raise TypeError(f"cannot read states from {item!r} ({type(item).__name__})")


def extract_channels(journeys: Iterable) -> list:
    """Unique marketing channels found in `journeys`, sorted alphabetically.

    `journeys` is an iterable of journeys; each journey is a path string, a list of states, or a
    single state name. Start, Conversion and Null are removed (they are structural states, not
    players, see the module docstring); duplicates are collapsed. The input is not modified.
    Zero channels is a legal result: it returns [].
    """
    channels = set()
    for item in journeys:
        for state in _states_of(item):
            if not state.strip():
                raise ValueError("empty state name found in journeys")
            if state not in STRUCTURAL_STATES:
                channels.add(state)
    return sorted(channels)


def extract_channels_from_journeys(journeys: pd.DataFrame) -> list:
    """Players from a journeys DataFrame (Person A's format: a `path` column of 'Start > ... > End').

    Every path is parsed with Person A's own `parse_path`, which also validates the structure
    (starts with Start, ends in Conversion/Null, no model state in the middle), so a malformed
    file fails loudly instead of yielding wrong players.
    """
    if "path" not in journeys.columns:
        raise ValueError("journeys DataFrame has no 'path' column")
    ids = journeys["journey_id"] if "journey_id" in journeys.columns else [None] * len(journeys)
    parsed = [parse_path(p, "?" if j is None else str(j)) for j, p in zip(ids, journeys["path"])]
    return extract_channels(parsed)


# --------------------------------------------------------------------------- 2. coalitions
def _check_channel_names(channels: Sequence) -> None:
    for c in channels:
        if not isinstance(c, str) or not c.strip():
            raise ValueError(f"invalid channel name: {c!r}")
        if c in STRUCTURAL_STATES or c.lower() in {s.lower() for s in STRUCTURAL_STATES}:
            raise CoalitionValidationError(f"{c!r} is a structural state, not a player")
        if COALITION_SEP in c:
            raise ValueError(f"channel name {c!r} contains the coalition separator {COALITION_SEP!r}")


def power_set_size(n_channels: int) -> int:
    """Number of coalitions for N players: 2^N."""
    if n_channels < 0:
        raise ValueError("n_channels must be >= 0")
    return 2 ** n_channels


def generate_coalitions(channels: Iterable, max_channels: int = DEFAULT_MAX_CHANNELS) -> list:
    """The complete power set of `channels` as a list of frozensets, in canonical order.

    Duplicates in `channels` are collapsed (players form a set) and input order does not matter:
    the result depends only on the SET of channels. The input is never modified. Raises
    CoalitionLimitError if N > max_channels (checked BEFORE generating anything).
    """
    original = list(channels)
    _check_channel_names(original)
    players = sorted(set(original))
    n = len(players)
    if n > max_channels:
        raise CoalitionLimitError(
            f"The number of channels produces an impractically large power set: {n} channels "
            f"-> 2^{n} = {power_set_size(n):,} coalitions (limit: max_channels={max_channels}, "
            f"i.e. {power_set_size(max_channels):,}). Raise max_channels only if you really need it."
        )
    # combinations() of a sorted list yields each size's subsets in lexicographic order.
    result = [frozenset(c) for k in range(n + 1) for c in combinations(players, k)]
    if isinstance(channels, Sequence) and list(channels) != original:
        raise CoalitionValidationError("the input channel collection was mutated")
    return result


def serialize_coalition(coalition: Iterable) -> str:
    """frozenset({'Google','Email'}) -> 'Email|Google'. Empty coalition -> '' (empty string)."""
    members = sorted(set(coalition))
    for m in members:
        if COALITION_SEP in m:
            raise ValueError(f"channel name {m!r} contains the coalition separator {COALITION_SEP!r}")
    return COALITION_SEP.join(members)


def deserialize_coalition(text: str) -> frozenset:
    """Inverse of `serialize_coalition`: 'Email|Google' -> frozenset; '' -> empty frozenset."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__} (was a NaN read from CSV? "
                        "use keep_default_na=False)")
    return frozenset(text.split(COALITION_SEP)) if text else frozenset()


def _sort_key(coalition: frozenset) -> tuple:
    return (len(coalition), tuple(sorted(coalition)))


# --------------------------------------------------------------------------- 3. validation
def validate_coalitions(coalitions: Sequence, channels: Iterable) -> None:
    """Raise CoalitionValidationError unless `coalitions` is exactly the canonical power set.

    Checks: N unique channels; 2^N coalitions; empty and grand coalition present; every member is
    a valid channel and never Start/Conversion/Null; no duplicates; correct number of coalitions of
    each size (C(N, k)); canonical deterministic order. Nothing is modified.
    """
    channel_list = list(channels)
    players = set(channel_list)
    n = len(players)
    if len(channel_list) != n:
        raise CoalitionValidationError("channel list contains duplicates; players must be unique")
    structural = {s.lower() for s in STRUCTURAL_STATES}
    bad = sorted(c for c in players if c.lower() in structural)
    if bad:
        raise CoalitionValidationError(f"structural state(s) used as players: {bad}")

    expected = power_set_size(n)
    if len(coalitions) != expected:
        raise CoalitionValidationError(f"expected 2^{n} = {expected} coalitions, got {len(coalitions)}")
    if not all(isinstance(c, frozenset) for c in coalitions):
        raise CoalitionValidationError("every coalition must be a frozenset")
    if len(set(coalitions)) != len(coalitions):
        raise CoalitionValidationError("duplicate coalitions found")
    if frozenset() not in set(coalitions):
        raise CoalitionValidationError("the empty coalition is missing")
    if frozenset(players) not in set(coalitions):
        raise CoalitionValidationError("the grand coalition is missing")
    for c in coalitions:
        outside = c - players
        if outside:
            raise CoalitionValidationError(f"coalition {sorted(c)} contains non-players: {sorted(outside)}")
    sizes = [len(c) for c in coalitions]
    for k in range(n + 1):
        if sizes.count(k) != comb(n, k):
            raise CoalitionValidationError(
                f"expected C({n},{k}) = {comb(n, k)} coalitions of size {k}, got {sizes.count(k)}")
    keys = [_sort_key(c) for c in coalitions]
    if keys != sorted(keys):
        raise CoalitionValidationError("coalitions are not in the canonical (size, names) order")


# --------------------------------------------------------------------------- 4. output
def coalitions_to_frame(coalitions: Sequence) -> pd.DataFrame:
    """Coalitions -> DataFrame[coalition_id, coalition_size, coalition(str)], ids 0..2^N-1."""
    return pd.DataFrame({
        "coalition_id": range(len(coalitions)),
        "coalition_size": [len(c) for c in coalitions],
        "coalition": [serialize_coalition(c) for c in coalitions],
    })[COALITION_COLUMNS]


def save_coalitions(coalitions: Sequence, path=DEFAULT_COALITIONS_PATH) -> Path:
    """Write coalitions.csv (creating the folder). The empty coalition is an empty string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coalitions_to_frame(coalitions).to_csv(path, index=False)
    return path


def load_coalitions(path=DEFAULT_COALITIONS_PATH) -> list:
    """Read coalitions.csv back into a list of frozensets (inverse of `save_coalitions`)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/run_stage6.py` first.")
    df = pd.read_csv(path, dtype={"coalition": str}, keep_default_na=False)
    missing = [c for c in COALITION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"coalitions file is missing required column(s): {missing}")
    if (df["coalition_size"] != df["coalition"].map(lambda s: len(deserialize_coalition(s)))).any():
        raise CoalitionValidationError("coalition_size does not match the coalition column")
    return [deserialize_coalition(s) for s in df["coalition"]]


def verify_saved_coalitions(path, channels: Iterable) -> None:
    """Re-read the CSV and validate it: proves what is on disk, not just what was in memory."""
    validate_coalitions(load_coalitions(path), channels)


# --------------------------------------------------------------------------- orchestration
def build_and_save(journeys_path=DEFAULT_OUTPUT_PATH, output_path=DEFAULT_COALITIONS_PATH,
                   max_channels: int = DEFAULT_MAX_CHANNELS) -> dict:
    """journeys.csv -> channels -> coalitions -> validate -> save -> re-read and verify."""
    journeys = load_journeys(journeys_path)
    channels = extract_channels_from_journeys(journeys)
    coalitions = generate_coalitions(channels, max_channels=max_channels)
    validate_coalitions(coalitions, channels)
    out = save_coalitions(coalitions, output_path)
    verify_saved_coalitions(out, channels)
    return {
        "n_journeys": len(journeys),
        "channels": channels,
        "n_channels": len(channels),
        "n_coalitions": len(coalitions),
        "expected_coalitions": power_set_size(len(channels)),
        "output_path": out,
    }


def main() -> None:
    r = build_and_save()
    print(f"Number of journeys:          {r['n_journeys']}")
    print(f"Number of unique channels:   {r['n_channels']}")
    print(f"Channels (players):          {', '.join(r['channels']) if r['channels'] else '(none)'}")
    print(f"Number of possible coalitions: {r['n_coalitions']}")
    print(f"Expected coalitions (2^N):   {r['expected_coalitions']}")
    print("Validation: PASSED (in memory and re-read from disk)")
    print(f"Output: {r['output_path']}")


if __name__ == "__main__":
    main()
