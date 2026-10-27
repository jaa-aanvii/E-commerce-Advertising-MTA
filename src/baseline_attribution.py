"""Heuristic reference attribution: first-touch, last-touch, linear.

Scope: ONLY the three position-based rules used as reference points against the
Markov removal effect and Shapley value (see hybrid_engine.py, budget_allocator.py).
Nothing here is a claim about which channel actually mattered - these rules credit
a touch by its POSITION in the journey, not by any estimated contribution.

Definitions (converting journeys only; a journey's path is "Start > c1 > ... > ck >
Conversion" from journey_builder.py - the eligible touches are c1..ck)
  First-touch : 1 unit of credit to c1.
  Last-touch  : 1 unit of credit to ck (the touch that carries the conversion,
                under journey_builder's assumption that the conversion event is
                the journey's final touch).
  Linear      : 1/k credit to EACH of c1..ck, so a channel visited twice in the
                same journey receives 2/k from that journey.

Non-converting (Null) journeys contribute no credit under any of the three rules:
there is no "the" touch to credit on a path that never converted. This differs
from the Markov/Shapley engines, which use every journey (see markov_engine.py).

Each rule's per-channel credits are summed across all converting journeys, then
normalized so the channel shares sum to 1 (ties-safe: every converting journey
contributes exactly 1.0 total credit under every rule, so the shares are exactly
comparable across rules and across engines).
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from mta.journey_builder import CONVERSION, PATH_SEP, START

RESULT_COLUMNS = ["channel", "credits", "weight", "percentage"]


def _eligible_touches(path: str) -> list:
    """'Start > A > B > Conversion' -> ['A', 'B'] (the credited touches)."""
    states = path.split(PATH_SEP)
    if states[0] != START or states[-1] != CONVERSION:
        raise ValueError(f"not a converting journey path: {path!r}")
    return states[1:-1]


def _credits_to_frame(credits: Counter) -> pd.DataFrame:
    total = sum(credits.values())
    if total <= 0:
        raise ValueError("no credit was assigned; are there any converting journeys?")
    rows = sorted(credits.items())
    table = pd.DataFrame(rows, columns=["channel", "credits"])
    table["weight"] = table["credits"] / total
    table["percentage"] = table["weight"] * 100.0
    return table[RESULT_COLUMNS]


def first_touch_attribution(journeys: pd.DataFrame) -> pd.DataFrame:
    """One credit per converting journey, to its first eligible touch."""
    credits: Counter = Counter()
    converting = journeys[journeys["converted"] == 1]
    for path in converting["path"]:
        touches = _eligible_touches(path)
        if touches:
            credits[touches[0]] += 1.0
    return _credits_to_frame(credits)


def last_touch_attribution(journeys: pd.DataFrame) -> pd.DataFrame:
    """One credit per converting journey, to its last eligible touch (the closer)."""
    credits: Counter = Counter()
    converting = journeys[journeys["converted"] == 1]
    for path in converting["path"]:
        touches = _eligible_touches(path)
        if touches:
            credits[touches[-1]] += 1.0
    return _credits_to_frame(credits)


def linear_attribution(journeys: pd.DataFrame) -> pd.DataFrame:
    """1/k credit per converting journey to EACH of its k eligible touches."""
    credits: Counter = Counter()
    converting = journeys[journeys["converted"] == 1]
    for path in converting["path"]:
        touches = _eligible_touches(path)
        if touches:
            share = 1.0 / len(touches)
            for ch in touches:
                credits[ch] += share
    return _credits_to_frame(credits)


def all_heuristic_attributions(journeys: pd.DataFrame) -> dict:
    """{'first': df, 'last': df, 'linear': df}, each from the same converting journeys."""
    return {
        "first": first_touch_attribution(journeys),
        "last": last_touch_attribution(journeys),
        "linear": linear_attribution(journeys),
    }
