"""Stage 11 tests.

The DataFrames built by markov() and shapley() below are small TEST FIXTURES with hand-chosen
numbers so every expected value can be checked on paper. They never touch project outputs. The
final test runs on the project's real Markov and Shapley CSVs when both are present.

Hand-solved base example (channels A and B):
  Markov RE  A=0.30  B=0.10  -> weights 0.75, 0.25
  Shapley    A=0.20  B=0.60  -> sum 0.80 (= baseline), weights 0.25, 0.75
  alpha=0 -> (0.25, 0.75)   alpha=1 -> (0.75, 0.25)   alpha=0.5 -> (0.50, 0.50)   alpha=0.8 -> (0.65, 0.35)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybrid_engine import (
    DEFAULT_MARKOV_PATH,
    DEFAULT_SHAPLEY_PATH,
    HYBRID_COLUMNS,
    cross_checks,
    export_hybrid,
    hybrid_attribution,
    load_markov_results,
    load_shapley_results,
    markov_weights,
    prepare_shapley,
    select_markov_segment,
    shapley_weights,
    validate_alpha,
)


def markov(effects, segment="all", baseline=0.8, status="ok"):
    return pd.DataFrame({
        "segment": segment,
        "channel": list(effects),
        "baseline_conversion_probability": baseline,
        "conversion_probability_without_channel": [baseline * (1 - e) for e in effects.values()],
        "removal_effect": list(effects.values()),
        "status": status,
        "n_visits": 10,
        "unreachable_states": "",
        "cut_off_states": "",
    })


def shapley(values):
    total = sum(values.values())
    return pd.DataFrame({
        "rank": range(1, len(values) + 1),
        "channel": list(values),
        "shapley_value": list(values.values()),
        "normalized_weight": [v / total for v in values.values()],
        "attribution_percentage": [100 * v / total for v in values.values()],
    })


BASE_M = {"A": 0.30, "B": 0.10}
BASE_S = {"A": 0.20, "B": 0.60}


def weights(table):
    return table.set_index("channel")["hybrid_weight"]


# ---------------------------------------------------------------- formula and endpoints
@pytest.mark.parametrize("alpha, a_weight", [(0, 0.25), (1, 0.75), (0.5, 0.50), (0.8, 0.65)])
def test_hybrid_matches_hand_calculation(alpha, a_weight):
    w = weights(hybrid_attribution(markov(BASE_M), shapley(BASE_S), alpha=alpha))
    assert w["A"] == pytest.approx(a_weight)
    assert w["B"] == pytest.approx(1 - a_weight)


def test_alpha_zero_equals_shapley_weights_and_alpha_one_equals_markov_weights():
    m, s = markov(BASE_M), shapley(BASE_S)
    at0 = hybrid_attribution(m, s, alpha=0).set_index("channel")
    at1 = hybrid_attribution(m, s, alpha=1).set_index("channel")
    assert at0["hybrid_weight"].to_dict() == pytest.approx(at0["shapley_weight"].to_dict())
    assert at1["hybrid_weight"].to_dict() == pytest.approx(at1["markov_weight"].to_dict())


def test_output_columns_and_percentage():
    t = hybrid_attribution(markov(BASE_M), shapley(BASE_S), alpha=0.5)
    assert list(t.columns) == HYBRID_COLUMNS
    assert (t["segment"] == "all").all() and (t["alpha"] == 0.5).all()
    assert t["hybrid_percentage"].to_numpy() == pytest.approx(100 * t["hybrid_weight"].to_numpy())


@pytest.mark.parametrize("alpha", [0, 0.25, 0.5, 0.75, 1])
def test_hybrid_weights_are_a_valid_distribution(alpha):
    t = hybrid_attribution(markov({"A": 0.3, "B": 0.1, "C": 0.2}), shapley({"A": 0.2, "B": 0.3, "C": 0.3}),
                           alpha=alpha)
    assert (t["hybrid_weight"] >= 0).all()
    assert t["hybrid_weight"].sum() == pytest.approx(1.0, abs=1e-12)
    assert t["hybrid_percentage"].sum() == pytest.approx(100.0)


# ---------------------------------------------------------------- alpha validation
@pytest.mark.parametrize("bad", [-0.1, 1.0001, float("nan"), float("inf"), float("-inf")])
def test_alpha_out_of_range_or_non_finite_raises_value_error(bad):
    with pytest.raises(ValueError):
        validate_alpha(bad)


@pytest.mark.parametrize("bad", ["0.5", None, True, [0.5]])
def test_alpha_of_wrong_type_raises_type_error(bad):
    with pytest.raises(TypeError):
        validate_alpha(bad)


@pytest.mark.parametrize("ok", [0, 1, 0.0, 1.0, 0.5, np.float64(0.3), np.int64(1)])
def test_alpha_boundaries_and_numpy_numbers_are_accepted(ok):
    assert 0 <= validate_alpha(ok) <= 1


def test_hybrid_rejects_bad_alpha():
    with pytest.raises(ValueError):
        hybrid_attribution(markov(BASE_M), shapley(BASE_S), alpha=1.5)


# ---------------------------------------------------------------- channel matching
def test_channels_are_matched_by_name_not_position_or_rank():
    m = markov(BASE_M)
    s = shapley(BASE_S).iloc[::-1].reset_index(drop=True)  # reverse the rows
    s["rank"] = [2, 1]                                      # and scramble rank
    w = weights(hybrid_attribution(m, s, alpha=0))
    assert w["A"] == pytest.approx(0.25) and w["B"] == pytest.approx(0.75)  # position matching would swap these


def test_channel_missing_from_either_method_raises_and_names_it():
    with pytest.raises(ValueError, match=r"Only in Markov: \['B'\]"):
        hybrid_attribution(markov(BASE_M), shapley({"A": 0.8}))
    with pytest.raises(ValueError, match=r"Only in Shapley: \['C'\]"):
        hybrid_attribution(markov(BASE_M), shapley({"A": 0.2, "B": 0.4, "C": 0.2}))


def test_duplicate_channel_rows_raise():
    with pytest.raises(ValueError, match="duplicate"):
        hybrid_attribution(pd.concat([markov(BASE_M), markov({"A": 0.3})], ignore_index=True), shapley(BASE_S))
    dup = pd.concat([shapley(BASE_S), shapley({"A": 0.2})], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        hybrid_attribution(markov(BASE_M), dup)


@pytest.mark.parametrize("state", ["Start", "Conversion", "Null", "null"])
def test_model_states_cannot_appear_as_channels(state):
    with pytest.raises(ValueError, match="model state"):
        hybrid_attribution(markov({"A": 0.3, state: 0.1}), shapley({"A": 0.2, state: 0.6}))


def test_channel_names_must_match_exactly():
    with pytest.raises(ValueError, match="channel sets differ"):
        hybrid_attribution(markov({"Email": 0.3, "B": 0.1}), shapley({"email": 0.2, "B": 0.6}))


# ---------------------------------------------------------------- invalid numbers / schema
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc"])
def test_non_finite_or_non_numeric_markov_effect_raises(bad):
    m = markov(BASE_M)
    m["removal_effect"] = m["removal_effect"].astype(object)
    m.loc[0, "removal_effect"] = bad
    with pytest.raises(ValueError, match="non-finite"):
        hybrid_attribution(m, shapley(BASE_S))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_shapley_value_raises(bad):
    s = shapley(BASE_S)
    s.loc[0, "shapley_value"] = bad
    with pytest.raises(ValueError, match="non-finite"):
        hybrid_attribution(markov(BASE_M), s, check_consistency=False)


def test_negative_markov_effect_raises():
    with pytest.raises(ValueError, match="non-negative"):
        hybrid_attribution(markov({"A": 0.3, "B": -0.1}), shapley(BASE_S), check_consistency=False)


def test_missing_columns_raise(tmp_path):
    (tmp_path / "m.csv").write_text("segment,channel\nall,A\n")
    (tmp_path / "s.csv").write_text("channel\nA\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_markov_results(tmp_path / "m.csv")
    with pytest.raises(ValueError, match="missing required column"):
        load_shapley_results(tmp_path / "s.csv")


def test_missing_files_have_helpful_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_stage5"):
        load_markov_results(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="DEFAULT_SHAPLEY_PATH"):
        load_shapley_results(tmp_path / "nope.csv")


# ---------------------------------------------------------------- Markov segments and statuses
def test_missing_markov_segment_raises_and_lists_available():
    with pytest.raises(ValueError, match=r"segment 'new' not available.*\['all'\]"):
        select_markov_segment(markov(BASE_M), "new")


def test_segment_selection_picks_only_that_segments_rows():
    both = pd.concat([markov(BASE_M, "all"), markov({"A": 0.5, "B": 0.1}, "new")], ignore_index=True)
    sel = select_markov_segment(both, "new").set_index("channel")
    assert sel.loc["A", "removal_effect"] == 0.5
    t = hybrid_attribution(both, shapley(BASE_S), segment="new", check_consistency=False)
    assert (t["segment"] == "new").all()


@pytest.mark.parametrize("status", ["zero_baseline", "not_in_journeys", "not_reachable_at_baseline"])
def test_non_ok_status_raises_instead_of_dropping_the_channel(status):
    m = markov(BASE_M)
    m.loc[1, "status"] = status
    with pytest.raises(ValueError, match=status):
        hybrid_attribution(m, shapley(BASE_S))


# ---------------------------------------------------------------- negative Shapley values
NEG_S = {"A": 0.5, "B": 0.3, "C": -0.1}
NEG_M = {"A": 0.2, "B": 0.2, "C": 0.1}


def test_negative_shapley_is_clipped_for_weights_but_raw_value_is_preserved():
    t = hybrid_attribution(markov(NEG_M, baseline=0.7), shapley(NEG_S), alpha=0.5).set_index("channel")
    assert t.loc["C", "shapley_raw_value"] == pytest.approx(-0.1)      # raw untouched, still negative
    assert bool(t.loc["C", "shapley_clipped"]) is True
    assert not t.loc[["A", "B"], "shapley_clipped"].any()
    assert t.loc["C", "shapley_weight"] == 0.0
    assert t.loc["A", "shapley_weight"] == pytest.approx(0.5 / 0.8)    # normalised by the positive part only
    assert t.loc["B", "shapley_weight"] == pytest.approx(0.3 / 0.8)
    assert t["shapley_weight"].sum() == pytest.approx(1.0)
    assert t["hybrid_weight"].sum() == pytest.approx(1.0)
    assert (t["hybrid_weight"] >= 0).all()
    assert t.loc["C", "hybrid_weight"] == pytest.approx(0.5 * 0.2 + 0.5 * 0.0)


def test_shapley_weights_function_directly():
    w, clipped = shapley_weights(pd.Series([0.5, 0.3, -0.1], index=["A", "B", "C"]))
    assert w.tolist() == pytest.approx([0.625, 0.375, 0.0])
    assert clipped.tolist() == [False, False, True]
    w2, c2 = shapley_weights(pd.Series([0.2, 0.6]))
    assert w2.tolist() == pytest.approx([0.25, 0.75]) and not c2.any()  # all positive: plain raw / sum


def test_shapley_weights_leave_the_input_unchanged():
    raw = pd.Series([0.5, -0.1])
    before = raw.copy()
    shapley_weights(raw)
    pd.testing.assert_series_equal(raw, before)


# ---------------------------------------------------------------- zero-sum / invalid distributions
def test_all_non_positive_shapley_values_raise():
    with pytest.raises(ValueError, match="no valid allocation weights"):
        shapley_weights(pd.Series([-0.2, -0.1, 0.0]))
    with pytest.raises(ValueError, match="no valid allocation weights"):
        hybrid_attribution(markov(BASE_M), shapley({"A": -0.2, "B": -0.1}) .assign(normalized_weight=0.0),
                           check_consistency=False)


def test_zero_sum_markov_effects_raise():
    with pytest.raises(ValueError, match="sum of removal effects is zero"):
        markov_weights(pd.Series([0.0, 0.0]))
    with pytest.raises(ValueError, match="sum of removal effects is zero"):
        hybrid_attribution(markov({"A": 0.0, "B": 0.0}), shapley(BASE_S), check_consistency=False)


def test_markov_weights_sum_to_one_even_though_effects_do_not():
    effects = pd.Series([0.4, 0.5, 0.5])  # sums to 1.4, like real removal effects
    assert effects.sum() > 1
    assert markov_weights(effects).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- raw preservation, determinism
def test_raw_scores_are_preserved_exactly():
    m, s = markov({"A": 0.3, "B": 0.1, "C": 0.2}), shapley({"A": 0.25, "B": 0.35, "C": 0.2})
    t = hybrid_attribution(m, s, alpha=0.3).set_index("channel")
    for ch, effect in zip(m["channel"], m["removal_effect"]):
        assert t.loc[ch, "markov_removal_effect"] == effect
    for ch, value in zip(s["channel"], s["shapley_value"]):
        assert t.loc[ch, "shapley_raw_value"] == value
    assert not np.allclose(t["markov_removal_effect"], t["markov_weight"])  # weights are a separate column


def test_inputs_are_not_modified():
    m, s = markov(BASE_M), shapley(BASE_S)
    m0, s0 = m.copy(), s.copy()
    hybrid_attribution(m, s, alpha=0.4)
    pd.testing.assert_frame_equal(m, m0)
    pd.testing.assert_frame_equal(s, s0)


def test_results_are_deterministic_and_independent_of_input_row_order():
    m, s = markov({"A": 0.3, "B": 0.1, "C": 0.2}), shapley({"A": 0.25, "B": 0.35, "C": 0.2})
    first = hybrid_attribution(m, s, alpha=0.37)
    again = hybrid_attribution(m, s, alpha=0.37)
    shuffled = hybrid_attribution(m.sample(frac=1, random_state=1), s.sample(frac=1, random_state=2), alpha=0.37)
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first, shuffled)


def test_ties_are_ordered_alphabetically():
    t = hybrid_attribution(markov({"B": 0.2, "A": 0.2}), shapley({"B": 0.4, "A": 0.4}))
    assert t["channel"].tolist() == ["A", "B"]


# ---------------------------------------------------------------- same journeys / consistent definitions
def test_baseline_consistency_check_passes_when_shapley_sums_to_the_markov_baseline():
    c = cross_checks(select_markov_segment(markov(BASE_M), "all"), prepare_shapley(shapley(BASE_S)))
    assert c["markov_baseline"] == pytest.approx(0.8) and c["shapley_sum"] == pytest.approx(0.8)
    assert c["baseline_consistent"] is True and c["existing_weights_match"] is True


def test_inconsistent_baselines_are_refused_by_default():
    with pytest.raises(ValueError, match="different journeys"):
        hybrid_attribution(markov(BASE_M, baseline=0.5), shapley(BASE_S))


def test_consistency_check_can_be_disabled_explicitly():
    t = hybrid_attribution(markov(BASE_M, baseline=0.5), shapley(BASE_S), check_consistency=False)
    assert t["hybrid_weight"].sum() == pytest.approx(1.0)


def test_cross_check_reports_a_normalised_weight_column_that_disagrees_with_raw_over_sum():
    s = shapley(BASE_S)
    s["normalized_weight"] = [0.5, 0.5]  # tampered
    c = cross_checks(select_markov_segment(markov(BASE_M), "all"), prepare_shapley(s))
    assert c["existing_weights_match"] is False and c["existing_weight_max_diff"] > 0.2


def test_shapley_file_without_segment_column_cannot_pass_for_a_different_segment():
    both = pd.concat([markov(BASE_M, "all"), markov({"A": 0.3, "B": 0.1}, "new", baseline=0.7)], ignore_index=True)
    with pytest.raises(ValueError, match="different journeys"):
        hybrid_attribution(both, shapley(BASE_S), segment="new")


# ---------------------------------------------------------------- export
def test_export_round_trip(tmp_path):
    t = hybrid_attribution(markov(BASE_M), shapley(BASE_S), alpha=0.5)
    out = export_hybrid(t, tmp_path / "nested" / "hybrid.csv")
    back = pd.read_csv(out)
    assert list(back.columns) == HYBRID_COLUMNS
    assert back["hybrid_weight"].sum() == pytest.approx(1.0)
    assert np.isfinite(back.select_dtypes("number").to_numpy()).all()


def test_markov_csv_with_blank_diagnostic_cells_loads(tmp_path):
    p = tmp_path / "m.csv"
    markov(BASE_M).to_csv(p, index=False)  # unreachable_states / cut_off_states are blank
    assert len(load_markov_results(p)) == 2


# ---------------------------------------------------------------- the project's real outputs
REAL_MARKOV, REAL_SHAPLEY = Path(DEFAULT_MARKOV_PATH), Path(DEFAULT_SHAPLEY_PATH)
needs_real = pytest.mark.skipif(not (REAL_MARKOV.exists() and REAL_SHAPLEY.exists()),
                                reason="real Markov and/or Shapley CSV not found at the default paths")


@needs_real
@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_real_outputs_integrate_and_satisfy_constraints(alpha):
    m, s = load_markov_results(REAL_MARKOV), load_shapley_results(REAL_SHAPLEY)
    t = hybrid_attribution(m, s, alpha=alpha)
    assert len(t) == 6 and t["channel"].is_unique
    assert set(t["channel"]) == set(s["channel"])
    assert (t["hybrid_weight"] >= 0).all() and t["hybrid_weight"].sum() == pytest.approx(1.0)
    sel = select_markov_segment(m, "all").set_index("channel")
    if alpha == 0:
        assert t.set_index("channel")["hybrid_weight"].to_dict() == pytest.approx(
            t.set_index("channel")["shapley_weight"].to_dict())
    if alpha == 1:
        expected = (sel["removal_effect"] / sel["removal_effect"].sum()).to_dict()
        assert t.set_index("channel")["hybrid_weight"].to_dict() == pytest.approx(expected)
    c = cross_checks(select_markov_segment(m, "all"), prepare_shapley(s))
    assert c["baseline_consistent"] and c["existing_weights_match"]
