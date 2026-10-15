"""Stage 7 (Person B, Task 1.2): the coalition value function V(S).

Scope: ONLY V(S) for every coalition in coalitions.csv. NO marginal contributions, Shapley values,
hybrid attribution, baselines or budget allocation (later tasks).

Definition (the whole stage rests on this)
  V(S) = the probability of EVENTUALLY reaching Conversion from Start in Person A's Markov model
         when only the channels in S are kept and every channel outside S is removed with Person A's
         redirect-to-Null policy.

How a coalition is evaluated
  1. Take Person A's fitted baseline probability matrix P (`fit_markov_model(journeys).probs`).
  2. removed = all channels - S.
  3. For each removed channel k apply Person A's `remove_channel` (every transition INTO k is added
     to the Null column, k's column is zeroed, k's row becomes "go to Null"). Removing several
     channels one after another gives the same matrix as removing them all at once, because each
     step only moves a column into Null.
  4. V(S) = Person A's `conversion_probability` of the resulting matrix.
  Nothing is re-implemented: the same functions that produce the Stage 5 removal effects produce V.

Consequences (all checked in validation)
  * V(N) = baseline conversion probability. With nothing removed, the matrix is P itself and the
    same function is applied to it.
  * V(empty set) is whatever the model gives when every channel is removed. It is NOT forced to 0:
    a direct Start -> Conversion transition makes it legitimately positive.
  * Values are never clipped. A value outside [0, 1] (beyond rounding noise) raises an error.

What V(S) is NOT: revenue, spend, ROI, a share of journeys containing a channel, or a count of
conversions. The data has no reliable channel-level spend or revenue, so none is used; the
attribution is purely in terms of Markov conversion probability.

Output: data/processed/coalition_values.csv, rows in EXACTLY the order and with the same
coalition_id as coalitions.csv. Values are written at full floating-point precision.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from shapley_engine.coalition_generator import (
    COALITION_COLUMNS,
    DEFAULT_COALITIONS_PATH,
    deserialize_coalition,
    serialize_coalition,
    validate_coalitions,
)
from mta.journey_builder import DEFAULT_OUTPUT_PATH
from mta.markov_engine import (
    TOL,
    eligible_channels,
    conversion_probability,
    fit_markov_model,
    load_journeys,
    matrix_problems,
    remove_channel,
)

DEFAULT_VALUES_PATH = Path("data/processed/coalition_values.csv")
VALUE_COLUMNS = COALITION_COLUMNS + [
    "value", "removed_channels", "baseline_conversion_probability", "max_row_sum_error",
]


class CoalitionValueError(ValueError):
    """A coalition value, or the table of values, is invalid."""


# --------------------------------------------------------------------------- 1. input
def load_coalition_table(path=DEFAULT_COALITIONS_PATH) -> pd.DataFrame:
    """Read coalitions.csv (Task 1.1 output) as a DataFrame, keeping coalition_id.

    keep_default_na=False so the empty coalition stays '' instead of becoming NaN.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/run_stage6.py` first.")
    df = pd.read_csv(path, dtype={"coalition": str}, keep_default_na=False)
    missing = [c for c in COALITION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"coalitions file is missing required column(s): {missing}")
    return df[COALITION_COLUMNS]


def table_coalitions(table: pd.DataFrame) -> list:
    """Coalition column -> list of frozensets, in table order."""
    return [deserialize_coalition(s) for s in table["coalition"]]


# --------------------------------------------------------------------------- 2. V(S)
def coalition_probability_matrix(P: pd.DataFrame, coalition: Iterable) -> pd.DataFrame:
    """Person A's matrix with every channel OUTSIDE `coalition` removed (redirect-to-Null).

    Returns a new matrix; P is not modified. Raises if the coalition names a state that is not a
    channel of P.
    """
    channels = eligible_channels(P)
    members = frozenset(coalition)
    unknown = sorted(members - set(channels))
    if unknown:
        raise CoalitionValueError(f"coalition contains non-channel(s): {unknown}")
    out = P
    for channel in channels:  # channels is in Person A's deterministic order
        if channel not in members:
            out = remove_channel(out, channel)
    return out


def calculate_coalition_value(P: pd.DataFrame, coalition: Iterable) -> float:
    """V(S): probability of eventually reaching Conversion from Start with only S's channels kept."""
    value, _ = _value_and_error(P, coalition)
    return value


def _value_and_error(P: pd.DataFrame, coalition: Iterable) -> tuple:
    """(V(S), largest |row sum - 1| of the coalition's matrix). Raises on any invalid matrix/value."""
    coalition = frozenset(coalition)
    P_s = coalition_probability_matrix(P, coalition)
    problems = matrix_problems(P_s)
    if problems:
        raise CoalitionValueError(f"coalition {sorted(coalition)}: invalid matrix: {problems}")
    row_error = float(np.abs(P_s.sum(axis=1).to_numpy(dtype=float) - 1.0).max())
    value = conversion_probability(P_s)
    if not np.isfinite(value) or value < -TOL or value > 1 + TOL:
        raise CoalitionValueError(f"coalition {sorted(coalition)}: value {value!r} is not a probability")
    return value + 0.0, row_error  # + 0.0 turns -0.0 into 0.0; no clipping


def calculate_all_coalition_values(P: pd.DataFrame, coalition_table: pd.DataFrame) -> pd.DataFrame:
    """One V(S) per row of `coalition_table`, in the same order and with the same coalition_id.

    The grand coalition is the set of all channels of P. The baseline column is Person A's
    conversion probability of the unmodified matrix.
    """
    channels = eligible_channels(P)
    baseline = conversion_probability(P)
    rows = []
    for cid, size, text in zip(coalition_table["coalition_id"], coalition_table["coalition_size"],
                               coalition_table["coalition"]):
        members = deserialize_coalition(text)
        value, row_error = _value_and_error(P, members)
        rows.append({
            "coalition_id": int(cid),
            "coalition_size": int(size),
            "coalition": text,
            "value": value,
            "removed_channels": serialize_coalition(set(channels) - members),
            "baseline_conversion_probability": baseline,
            "max_row_sum_error": row_error,
        })
    return pd.DataFrame(rows, columns=VALUE_COLUMNS)


# --------------------------------------------------------------------------- 3. validation
def validate_coalition_values(values: pd.DataFrame, coalition_table: pd.DataFrame, channels: Iterable,
                              baseline: float, tol: float = 1e-9) -> None:
    """Raise CoalitionValueError unless `values` is a correct V(S) table for `coalition_table`.

    Checks: required columns; no duplicate coalition_id; no missing/extra coalition; one value per
    coalition and same count; same order, size and coalition text as coalitions.csv; members are
    valid channels; empty and grand coalition present; value numeric, not NaN, finite, within [0, 1]
    (no clipping); V(N) == baseline; every coalition matrix's rows sum to 1 (max_row_sum_error).
    """
    channels = set(channels)
    missing_cols = [c for c in VALUE_COLUMNS if c not in values.columns]
    if missing_cols:
        raise CoalitionValueError(f"values table is missing column(s): {missing_cols}")
    if coalition_table["coalition_id"].duplicated().any():
        raise CoalitionValueError("coalitions table has duplicate coalition_id values")
    dup = values.loc[values["coalition_id"].duplicated(), "coalition_id"].tolist()
    if dup:
        raise CoalitionValueError(f"duplicate coalition_id in values: {sorted(set(dup))}")
    expected_ids = set(coalition_table["coalition_id"])
    got_ids = set(values["coalition_id"])
    if expected_ids - got_ids:
        raise CoalitionValueError(f"missing value for coalition_id(s): {sorted(expected_ids - got_ids)}")
    if got_ids - expected_ids:
        raise CoalitionValueError(f"values for unknown coalition_id(s): {sorted(got_ids - expected_ids)}")
    if len(values) != len(coalition_table):
        raise CoalitionValueError(f"{len(values)} values for {len(coalition_table)} coalitions")
    for col in COALITION_COLUMNS:
        if values[col].tolist() != coalition_table[col].tolist():
            raise CoalitionValueError(f"column {col!r} differs from coalitions.csv (order must be identical)")

    coalitions = table_coalitions(values)
    for c in coalitions:
        if c - channels:
            raise CoalitionValueError(f"coalition {sorted(c)} contains non-channels: {sorted(c - channels)}")
    if frozenset() not in set(coalitions):
        raise CoalitionValueError("the empty coalition is missing")
    if frozenset(channels) not in set(coalitions):
        raise CoalitionValueError("the grand coalition is missing")

    v = values["value"]
    if not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
        raise CoalitionValueError("values must be numeric")
    arr = v.to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise CoalitionValueError("some values are NaN")
    if not np.isfinite(arr).all():
        raise CoalitionValueError("some values are infinite")
    if (arr < -tol).any() or (arr > 1 + tol).any():
        raise CoalitionValueError("some values are outside [0, 1]; they are probabilities and are never clipped")

    grand_value = float(v[[c == frozenset(channels) for c in coalitions]].iloc[0])
    if abs(grand_value - baseline) > tol:
        raise CoalitionValueError(f"V(N) = {grand_value} differs from baseline conversion probability {baseline}")
    if (values["baseline_conversion_probability"].astype(float) - baseline).abs().max() > tol:
        raise CoalitionValueError("baseline_conversion_probability column disagrees with the baseline")
    if values["max_row_sum_error"].astype(float).max() > tol:
        raise CoalitionValueError("some coalition matrices have rows that do not sum to 1")


# --------------------------------------------------------------------------- 4. output
def save_coalition_values(values: pd.DataFrame, path=DEFAULT_VALUES_PATH) -> Path:
    """Write coalition_values.csv (creating the folder). Full float precision, no rounding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values[VALUE_COLUMNS].to_csv(path, index=False)
    return path


def load_coalition_values(path=DEFAULT_VALUES_PATH) -> pd.DataFrame:
    """Read coalition_values.csv back (empty coalition / removed_channels stay '')."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/run_stage7.py` first.")
    return pd.read_csv(path, dtype={"coalition": str, "removed_channels": str}, keep_default_na=False)


# --------------------------------------------------------------------------- orchestration
def build_and_save(journeys_path=DEFAULT_OUTPUT_PATH, coalitions_path=DEFAULT_COALITIONS_PATH,
                   output_path=DEFAULT_VALUES_PATH) -> dict:
    """journeys.csv + coalitions.csv -> V(S) for every coalition -> validate -> save -> re-read, validate."""
    journeys = load_journeys(journeys_path)
    model = fit_markov_model(journeys)  # Person A's model on ALL journeys
    P = model.probs
    channels = eligible_channels(P)
    table = load_coalition_table(coalitions_path)
    # Task 1.1's validator: coalitions.csv must be exactly the power set of THIS model's channels.
    if table["coalition_id"].tolist() != list(range(len(table))):
        raise CoalitionValueError("coalition_id must be 0..2^N-1 in order")
    validate_coalitions(table_coalitions(table), channels)
    values = calculate_all_coalition_values(P, table)
    baseline = model.base_conversion_probability
    validate_coalition_values(values, table, channels, baseline)
    out = save_coalition_values(values, output_path)
    validate_coalition_values(load_coalition_values(out), table, channels, baseline)
    is_grand = values["removed_channels"] == ""
    return {
        "n_channels": len(channels),
        "channels": channels,
        "n_coalitions": len(table),
        "expected_coalitions": 2 ** len(channels),
        "n_values": len(values),
        "baseline": baseline,
        "grand_value": float(values.loc[is_grand, "value"].iloc[0]),
        "empty_value": float(values.loc[values["coalition_size"] == 0, "value"].iloc[0]),
        "output_path": out,
    }


def main() -> None:
    r = build_and_save()
    print(f"Number of unique channels:         {r['n_channels']}")
    print(f"Number of coalitions:              {r['n_coalitions']}")
    print(f"Expected coalitions (2^N):         {r['expected_coalitions']}")
    print(f"Number of coalition values:        {r['n_values']}")
    print(f"Baseline conversion probability:   {r['baseline']:.10f}")
    print(f"V(grand coalition):                {r['grand_value']:.10f}")
    print(f"V(empty coalition):                {r['empty_value']:.10f}")
    print(f"V(N) == baseline:                  {abs(r['grand_value'] - r['baseline']) <= 1e-9}")
    print("Validation: PASSED (in memory and re-read from disk)")
    print(f"Output: {r['output_path']}")


if __name__ == "__main__":
    main()
