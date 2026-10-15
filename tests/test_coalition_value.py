"""Stage 7 / Task 1.2 tests: coalition value V(S) = Markov conversion probability with the channels
outside S redirected to Null.

Every journey list below is a hand-calculable TEST FIXTURE, invented only to isolate one rule. It
never feeds any project result. The last test runs on the real files and is skipped if they are
missing. Hand calculations are written next to each fixture.
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import coalition_value
from coalition_generator import generate_coalitions, save_coalitions
from coalition_value import (
    CoalitionValueError,
    build_and_save,
    calculate_all_coalition_values,
    calculate_coalition_value,
    coalition_probability_matrix,
    load_coalition_table,
    load_coalition_values,
    save_coalition_values,
    validate_coalition_values,
)
from mta.journey_builder import DEFAULT_OUTPUT_PATH
from mta.markov_engine import conversion_probability, conversion_probability_iterative, fit_markov_model, matrix_problems

TOL = 1e-9


def journeys_frame(paths):
    return pd.DataFrame({"journey_id": [f"u{i}#1" for i in range(len(paths))], "path": paths,
                         "segment": "new", "converted": [int(p.endswith("Conversion")) for p in paths]})


# Fixture ONE: Start > A > Conversion only.  P(Start->A)=1, P(A->Conv)=1.  V({A})=1, V(empty)=0.
ONE = ["Start > A > Conversion"] * 3

# Fixture TWO (parallel channels): Start->A .5, Start->B .5; A->Conv 1; B->Conv .5, B->Null .5.
#   baseline = .5*1 + .5*.5 = .75     V({A}) = .5     V({B}) = .25     V(empty) = 0
TWO = ["Start > A > Conversion", "Start > A > Conversion", "Start > B > Conversion", "Start > B > Null"]

# Fixture THREE (chained + direct): Start->A .5, Start->B .25, Start->Conv .25; A->B .5, A->Null .5;
#   B->Conv 1.   baseline = .25 + .5*.5*1 + .25*1 = .75
#   V({A}): B removed, so A->B becomes A->Null and Start->B becomes Start->Null: V = .25 (direct only)
#   V({B}): A removed, Start->A becomes Null: V = .25 direct + .25 via B = .5
#   V(empty) = .25  (the direct Start->Conversion transition survives: NOT forced to 0)
THREE = ["Start > A > B > Conversion", "Start > A > Null", "Start > B > Conversion", "Start > Conversion"]


def model_P(paths):
    return fit_markov_model(journeys_frame(paths)).probs


def full_table(channels):
    """coalitions.csv-shaped table straight from Task 1.1's generator (ids 0..2^N-1)."""
    from coalition_generator import coalitions_to_frame
    return coalitions_to_frame(generate_coalitions(channels))


def value_of(values, coalition_text):
    return float(values.loc[values["coalition"] == coalition_text, "value"].iloc[0])


# ---------------------------------------------------------------- 1, 8: grand coalition == baseline
def test_01_all_channels_give_baseline_conversion_probability():
    for paths in (ONE, TWO, THREE):
        P = model_P(paths)
        assert calculate_coalition_value(P, {"A", "B"} & set(P.index)) == pytest.approx(conversion_probability(P), abs=TOL)


def test_08_grand_coalition_value_equals_baseline_in_the_table():
    P = model_P(TWO)
    values = calculate_all_coalition_values(P, full_table(["A", "B"]))
    assert value_of(values, "A|B") == pytest.approx(0.75, abs=TOL)
    assert values["baseline_conversion_probability"].iloc[0] == pytest.approx(0.75, abs=TOL)


# ---------------------------------------------------------------- 2, 9: empty coalition and hand examples
def test_02_empty_coalition_can_be_evaluated():
    P = model_P(TWO)
    assert calculate_coalition_value(P, set()) == pytest.approx(0.0, abs=TOL)


def test_09_hand_calculable_single_channel_example():
    P = model_P(ONE)
    assert calculate_coalition_value(P, {"A"}) == pytest.approx(1.0, abs=TOL)  # = baseline
    assert calculate_coalition_value(P, set()) == pytest.approx(0.0, abs=TOL)


def test_09b_hand_calculable_parallel_channels():
    values = calculate_all_coalition_values(model_P(TWO), full_table(["A", "B"]))
    assert value_of(values, "") == pytest.approx(0.0, abs=TOL)
    assert value_of(values, "A") == pytest.approx(0.5, abs=TOL)
    assert value_of(values, "B") == pytest.approx(0.25, abs=TOL)
    assert value_of(values, "A|B") == pytest.approx(0.75, abs=TOL)


def test_09c_hand_calculable_chain_and_nonzero_empty_coalition():
    values = calculate_all_coalition_values(model_P(THREE), full_table(["A", "B"]))
    assert value_of(values, "") == pytest.approx(0.25, abs=TOL)  # direct Start -> Conversion survives
    assert value_of(values, "A") == pytest.approx(0.25, abs=TOL)
    assert value_of(values, "B") == pytest.approx(0.5, abs=TOL)
    assert value_of(values, "A|B") == pytest.approx(0.75, abs=TOL)


def test_09d_value_agrees_with_independent_step_by_step_calculation():
    P = model_P(THREE)
    for c in generate_coalitions(["A", "B"]):
        P_s = coalition_probability_matrix(P, c)
        assert calculate_coalition_value(P, c) == pytest.approx(conversion_probability_iterative(P_s), abs=1e-9)


# ---------------------------------------------------------------- 3, 4: redirect-to-Null, normalisation
def test_03_channel_outside_coalition_is_redirected_to_null():
    P = model_P(THREE)
    P_s = coalition_probability_matrix(P, {"A"})  # B removed
    assert P_s["B"].abs().sum() == 0  # nothing can enter B any more
    assert P_s.loc["B", "Null"] == 1.0  # B itself is just absorbed into Null
    assert P_s.loc["Start", "Null"] == pytest.approx(P.loc["Start", "Null"] + P.loc["Start", "B"])
    assert P_s.loc["A", "Null"] == pytest.approx(P.loc["A", "Null"] + P.loc["A", "B"])
    assert P_s.loc["Start", "A"] == P.loc["Start", "A"]  # kept channels are untouched
    assert P_s.loc["Start", "Conversion"] == P.loc["Start", "Conversion"]


def test_03b_input_matrix_is_not_modified():
    P = model_P(THREE)
    before = P.copy()
    coalition_probability_matrix(P, set())
    pd.testing.assert_frame_equal(P, before)


def test_04_every_coalition_matrix_stays_normalised_and_absorbing():
    P = model_P(THREE)
    for c in generate_coalitions(["A", "B"]):
        P_s = coalition_probability_matrix(P, c)
        assert matrix_problems(P_s) == []
        assert np.allclose(P_s.sum(axis=1), 1.0, atol=TOL)
        assert P_s.loc["Conversion", "Conversion"] == 1.0 and P_s.loc["Null", "Null"] == 1.0
    values = calculate_all_coalition_values(P, full_table(["A", "B"]))
    assert (values["max_row_sum_error"] <= TOL).all()


def test_unknown_channel_in_coalition_is_rejected():
    with pytest.raises(CoalitionValueError, match="non-channel"):
        calculate_coalition_value(model_P(TWO), {"A", "Z"})


# ---------------------------------------------------------------- 5, 6, 7: table properties
def test_05_values_are_numeric_and_finite():
    values = calculate_all_coalition_values(model_P(THREE), full_table(["A", "B"]))
    assert pd.api.types.is_numeric_dtype(values["value"])
    assert np.isfinite(values["value"].to_numpy(dtype=float)).all()
    assert ((values["value"] >= -TOL) & (values["value"] <= 1 + TOL)).all()


def test_06_and_07_one_value_per_coalition_in_coalitions_csv_order(tmp_path):
    jp = tmp_path / "journeys.csv"
    journeys_frame(THREE).to_csv(jp, index=False)
    cp = save_coalitions(generate_coalitions(["B", "A"]), tmp_path / "coalitions.csv")
    r = build_and_save(jp, cp, tmp_path / "values.csv")
    table, values = load_coalition_table(cp), load_coalition_values(r["output_path"])
    assert len(values) == len(table) == r["n_coalitions"] == r["expected_coalitions"] == 4
    assert values["coalition_id"].is_unique
    assert values[["coalition_id", "coalition_size", "coalition"]].equals(table)  # same order, same ids
    assert values.loc[0, "coalition"] == "" and values.loc[0, "coalition_size"] == 0


def test_removed_channels_column_is_the_complement():
    values = calculate_all_coalition_values(model_P(TWO), full_table(["A", "B"]))
    assert dict(zip(values["coalition"], values["removed_channels"])) == {"": "A|B", "A": "B", "B": "A", "A|B": ""}


# ---------------------------------------------------------------- 10: no revenue / spend
def test_10_no_revenue_spend_or_roi_is_used():
    values = calculate_all_coalition_values(model_P(THREE), full_table(["A", "B"]))
    assert not any(w in c.lower() for c in values.columns for w in ("revenue", "spend", "roi", "cost"))
    assert "revenue" not in inspect.signature(calculate_coalition_value).parameters
    # Adding revenue-like columns to the journeys must not change a single value.
    noisy = journeys_frame(THREE).assign(revenue=[1e6, 5.0, 77.0, 0.0], spend=[9.0, 9.0, 9.0, 9.0])
    values2 = calculate_all_coalition_values(fit_markov_model(noisy).probs, full_table(["A", "B"]))
    assert values["value"].tolist() == values2["value"].tolist()


# ---------------------------------------------------------------- 11, 12 + other validation failures
def ok_case():
    P = model_P(THREE)
    table = full_table(["A", "B"])
    return P, table, calculate_all_coalition_values(P, table)


def validate(values, table):
    validate_coalition_values(values, table, ["A", "B"], baseline=0.75)


def test_validation_passes_on_correct_table():
    _, table, values = ok_case()
    validate(values, table)


def test_11_missing_coalition_value_is_detected():
    _, table, values = ok_case()
    with pytest.raises(CoalitionValueError, match="missing value"):
        validate(values.drop(index=2).reset_index(drop=True), table)


def test_12_duplicate_coalition_ids_are_detected():
    _, table, values = ok_case()
    dup = pd.concat([values, values.iloc[[1]]], ignore_index=True)
    with pytest.raises(CoalitionValueError, match="duplicate coalition_id"):
        validate(dup, table)


def test_validation_detects_wrong_order_and_extra_ids():
    _, table, values = ok_case()
    swapped = values.iloc[[1, 0, 2, 3]].reset_index(drop=True)
    with pytest.raises(CoalitionValueError, match="differs from coalitions.csv"):
        validate(swapped, table)
    extra = pd.concat([values, values.iloc[[0]].assign(coalition_id=99)], ignore_index=True)
    with pytest.raises(CoalitionValueError, match="unknown coalition_id"):
        validate(extra, table)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1.5, -0.2])
def test_validation_rejects_nan_infinite_and_out_of_range_values(bad):
    _, table, values = ok_case()
    values.loc[1, "value"] = bad
    with pytest.raises(CoalitionValueError):
        validate(values, table)


def test_validation_rejects_non_numeric_values():
    _, table, values = ok_case()
    values["value"] = values["value"].astype(str)
    with pytest.raises(CoalitionValueError, match="numeric"):
        validate(values, table)


def test_validation_rejects_grand_coalition_not_equal_to_baseline():
    _, table, values = ok_case()
    with pytest.raises(CoalitionValueError, match="baseline"):
        validate_coalition_values(values, table, ["A", "B"], baseline=0.5)


def test_validation_rejects_unnormalised_matrices_and_foreign_channels():
    _, table, values = ok_case()
    bad = values.copy()
    bad.loc[0, "max_row_sum_error"] = 0.1
    with pytest.raises(CoalitionValueError, match="sum to 1"):
        validate(bad, table)
    with pytest.raises(CoalitionValueError, match="non-channels"):
        validate_coalition_values(values, table, ["A"], baseline=0.75)


def test_out_of_range_value_raises_instead_of_being_clipped():
    P = model_P(TWO)
    original = coalition_value.conversion_probability
    try:
        for bad in (1.3, -0.4, float("nan")):
            coalition_value.conversion_probability = lambda _m, bad=bad: bad
            with pytest.raises(CoalitionValueError, match="not a probability"):
                calculate_coalition_value(P, {"A"})
    finally:
        coalition_value.conversion_probability = original


# ---------------------------------------------------------------- CSV and orchestration
def test_save_load_roundtrip_keeps_full_precision_and_empty_strings(tmp_path):
    _, table, values = ok_case()
    out = save_coalition_values(values, tmp_path / "x" / "v.csv")
    back = load_coalition_values(out)
    assert back.loc[0, "coalition"] == "" and back.loc[3, "removed_channels"] == ""
    assert back["value"].tolist() == values["value"].tolist()
    validate(back, table)


def test_build_and_save_with_zero_channels(tmp_path):
    jp = tmp_path / "j.csv"
    journeys_frame(["Start > Conversion", "Start > Null"]).to_csv(jp, index=False)
    cp = save_coalitions(generate_coalitions([]), tmp_path / "c.csv")
    r = build_and_save(jp, cp, tmp_path / "v.csv")
    assert r["n_values"] == 1
    assert r["empty_value"] == pytest.approx(0.5, abs=TOL)  # 1 of 2 journeys converts
    assert r["empty_value"] == r["baseline"] == r["grand_value"]  # the only coalition is both empty and grand


def test_build_and_save_rejects_coalitions_that_are_not_this_models_power_set(tmp_path):
    jp = tmp_path / "j.csv"
    journeys_frame(THREE).to_csv(jp, index=False)
    cp = save_coalitions(generate_coalitions(["A", "B", "C"]), tmp_path / "c.csv")
    with pytest.raises(ValueError):
        build_and_save(jp, cp, tmp_path / "v.csv")


def test_missing_input_files_fail_clearly(tmp_path):
    jp = tmp_path / "j.csv"
    journeys_frame(THREE).to_csv(jp, index=False)
    with pytest.raises(FileNotFoundError, match="run_stage6"):
        build_and_save(jp, tmp_path / "nope.csv", tmp_path / "v.csv")
    with pytest.raises(FileNotFoundError, match="run_stage3"):
        build_and_save(tmp_path / "nope.csv", tmp_path / "c.csv", tmp_path / "v.csv")


# ---------------------------------------------------------------- real data
def test_real_data_grand_coalition_equals_baseline(tmp_path):
    coalitions_path = Path("data/processed/coalitions.csv")
    if not (Path(DEFAULT_OUTPUT_PATH).exists() and coalitions_path.exists()):
        pytest.skip("journeys.csv / coalitions.csv not found; run scripts/run_stage3.py and run_stage6.py first")
    r = build_and_save(DEFAULT_OUTPUT_PATH, coalitions_path, tmp_path / "v.csv")
    assert r["n_values"] == r["expected_coalitions"] == 2 ** r["n_channels"]
    assert abs(r["grand_value"] - r["baseline"]) <= TOL