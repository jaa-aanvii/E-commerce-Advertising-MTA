"""Stage 5 tests.

EXAMPLE_PATHS are 9 REAL journeys from the Kaggle file (traced by hand in Stage 3). Every other
path list is a small TEST FIXTURE, invented only to isolate one rule. Fixtures never feed any
result. The last tests run on the real file.

Hand-derived answers for the 9-journey example (redirect-to-Null policy), baseline CR = 7/9:
  remove Search Ads    -> CR 4/9  -> RE 3/7        remove Email          -> CR 5/9  -> RE 2/7
  remove Social Media  -> CR 4/9  -> RE 3/7        remove Direct Traffic -> CR 6/9  -> RE 1/7
  remove Display Ads   -> CR 6/9  -> RE 1/7
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mta.data_pipeline import KAGGLE_CONFIG, clean_events
from mta.journey_builder import build_journeys
from mta.markov_engine import (
    CONVERSION,
    NULL,
    RESULT_COLUMNS,
    START,
    absorption_probabilities,
    all_removal_effects,
    build_count_matrix,
    build_probability_matrix,
    conversion_probability,
    count_transitions,
    eligible_channels,
    export_markov_results,
    fit_markov_model,
    fit_segment_models,
    matrix_problems,
    parse_path,
    reachable_states,
    remove_channel,
    removal_effect_from_probabilities,
    removal_effects,
)

EXAMPLE_PATHS = [
    "Start > Social Media > Direct Traffic > Email > Conversion",
    "Start > Social Media > Display Ads > Display Ads > Conversion",
    "Start > Search Ads > Conversion",
    "Start > Social Media > Social Media > Conversion",
    "Start > Search Ads > Conversion",
    "Start > Social Media > Null",
    "Start > Email > Conversion",
    "Start > Search Ads > Conversion",
    "Start > Social Media > Null",
]
HAND = {  # channel -> (CR_without, RE)
    "Search Ads": (4 / 9, 3 / 7),
    "Social Media": (4 / 9, 3 / 7),
    "Email": (5 / 9, 2 / 7),
    "Direct Traffic": (6 / 9, 1 / 7),
    "Display Ads": (6 / 9, 1 / 7),
}


def matrices(paths):
    counts = build_count_matrix(count_transitions([parse_path(p) for p in paths]))
    return build_probability_matrix(counts), counts


@pytest.fixture
def example():
    return matrices(EXAMPLE_PATHS)


# ---------------------------------------------------------------- manually solvable example
def test_removal_effects_match_hand_calculation(example):
    P, counts = example
    table = removal_effects(P, counts=counts).set_index("channel")
    assert set(table.index) == set(HAND)
    for channel, (cr_without, re) in HAND.items():
        row = table.loc[channel]
        assert row["baseline_conversion_probability"] == pytest.approx(7 / 9)
        assert row["conversion_probability_without_channel"] == pytest.approx(cr_without)
        assert row["removal_effect"] == pytest.approx(re)


def test_formula_is_one_minus_ratio():
    assert removal_effect_from_probabilities(0.8, 0.6) == pytest.approx(1 - 0.6 / 0.8)
    assert removal_effect_from_probabilities(0.5, 0.5) == 0.0
    assert removal_effect_from_probabilities(0.5, 0.0) == 1.0


def test_table_columns_order_and_ranking(example):
    P, counts = example
    table = removal_effects(P, counts=counts, segment="demo")
    assert list(table.columns) == RESULT_COLUMNS
    assert (table["segment"] == "demo").all()
    # highest effect first; exact ties are broken alphabetically (so output is reproducible)
    assert table["channel"].tolist() == ["Search Ads", "Social Media", "Email", "Direct Traffic", "Display Ads"]
    assert table["n_visits"].tolist() == [3, 6, 2, 1, 2]  # arrivals into each channel


# ---------------------------------------------------------------- the removal policy itself
def test_removal_redirects_incoming_mass_to_null(example):
    P, _ = example
    P2 = remove_channel(P, "Search Ads")
    assert P2.loc[START, "Search Ads"] == 0.0
    assert P2.loc[START, NULL] == pytest.approx(3 / 9)          # Search's share went to Null
    assert P2.loc[START, "Social Media"] == pytest.approx(5 / 9)  # other probabilities NOT rescaled
    assert P2.loc[START, "Email"] == pytest.approx(1 / 9)
    assert (P2["Search Ads"] == 0).all()                         # nothing reaches the removed channel


def test_rows_still_sum_to_one_after_removal(example):
    P, _ = example
    for ch in eligible_channels(P):
        P2 = remove_channel(P, ch)
        assert np.allclose(P2.sum(axis=1).to_numpy(), 1.0)
        assert matrix_problems(P2) == []


def test_self_loop_of_removed_channel_is_handled(example):
    P, _ = example
    P2 = remove_channel(P, "Display Ads")  # Display Ads has a self-loop and a route in from Social
    assert P2.loc["Social Media", NULL] == pytest.approx(2 / 6 + 1 / 6)  # old Null + redirected Display
    assert P2.loc["Display Ads", "Display Ads"] == 0.0
    assert P2.loc["Display Ads", NULL] == 1.0


def test_input_matrix_is_not_modified(example):
    P, _ = example
    before = P.copy()
    remove_channel(P, "Email")
    pd.testing.assert_frame_equal(P, before)


def test_naive_delete_and_renormalise_gives_a_different_answer(example):
    # Deleting Search's row/column leaves rows < 1. "Fixing" them by renormalising assumes users
    # switch to other channels: CR becomes 2/3 instead of 4/9, and RE falls from 3/7 to 1/7.
    P, _ = example
    naive = P.drop(index="Search Ads", columns="Search Ads")
    assert naive.loc[START].sum() < 1 - 1e-9  # the column deletion alone breaks the distribution
    for state in naive.index:
        if state not in (CONVERSION, NULL):
            naive.loc[state] = naive.loc[state] / naive.loc[state].sum()
    assert conversion_probability(naive) == pytest.approx(2 / 3)
    correct = removal_effects(P).set_index("channel").loc["Search Ads"]
    assert correct["conversion_probability_without_channel"] == pytest.approx(4 / 9)
    assert correct["conversion_probability_without_channel"] != pytest.approx(2 / 3)


def test_first_order_chain_can_recombine_paths_unlike_journey_replay():
    # Fixture. Journeys: S>A>B>C and S>B>N. Replaying journeys without A gives 0 conversions, but the
    # chain lets Start -> B -> Conversion happen (1/2 * 1/2 = 1/4). RE_A = 1/2, not 1.
    P, _ = matrices(["Start > A > B > Conversion", "Start > B > Null"])
    table = removal_effects(P).set_index("channel")
    assert table.loc["A", "baseline_conversion_probability"] == pytest.approx(0.5)
    assert table.loc["A", "conversion_probability_without_channel"] == pytest.approx(0.25)
    assert table.loc["A", "removal_effect"] == pytest.approx(0.5)
    assert table.loc["B", "removal_effect"] == pytest.approx(1.0)  # B is on every converting path


# ---------------------------------------------------------------- terminal states
@pytest.mark.parametrize("state", [START, CONVERSION, NULL])
def test_model_states_cannot_be_removed(example, state):
    P, _ = example
    with pytest.raises(ValueError, match="model state"):
        remove_channel(P, state)
    with pytest.raises(ValueError, match="model states"):
        removal_effects(P, channels=[state])


def test_terminal_states_stay_absorbing_after_removal(example):
    P, _ = example
    for ch in eligible_channels(P):
        P2 = remove_channel(P, ch)
        assert P2.loc[CONVERSION, CONVERSION] == 1.0 and P2.loc[CONVERSION].drop(CONVERSION).sum() == 0
        assert P2.loc[NULL, NULL] == 1.0 and P2.loc[NULL].drop(NULL).sum() == 0
        assert P2[START].sum() == 0


def test_eligible_channels_exclude_model_states(example):
    P, _ = example
    assert set(eligible_channels(P)) == set(HAND)


def test_removal_never_increases_conversion_and_effects_are_in_unit_interval(example):
    P, _ = example
    t = removal_effects(P)
    assert (t["conversion_probability_without_channel"] <= t["baseline_conversion_probability"] + 1e-12).all()
    assert t["removal_effect"].between(0, 1).all()


# ---------------------------------------------------------------- edge cases
def test_zero_baseline_gives_zero_effects_with_explicit_status():
    P, counts = matrices(["Start > A > Null", "Start > B > Null"])
    t = removal_effects(P, counts=counts)
    assert (t["baseline_conversion_probability"] == 0).all()
    assert (t["removal_effect"] == 0).all()
    assert (t["status"] == "zero_baseline").all()
    assert np.isfinite(t[["baseline_conversion_probability", "conversion_probability_without_channel",
                          "removal_effect"]].to_numpy()).all()


def test_zero_baseline_helper_does_not_divide_by_zero():
    assert removal_effect_from_probabilities(0.0, 0.0) == 0.0


def test_channel_absent_from_journeys_has_zero_effect(example):
    P, counts = example
    t = removal_effects(P, channels=list(HAND) + ["Referral"], counts=counts).set_index("channel")
    row = t.loc["Referral"]
    assert row["status"] == "not_in_journeys"
    assert row["removal_effect"] == 0.0
    assert row["conversion_probability_without_channel"] == pytest.approx(row["baseline_conversion_probability"])
    assert row["n_visits"] == 0
    assert t.loc["Search Ads", "status"] == "ok"  # the others are unaffected


def test_unreachable_states_are_reported(example):
    P, counts = example
    t = removal_effects(P, counts=counts).set_index("channel")
    # Direct Traffic and Display Ads are only reachable through Social Media.
    assert t.loc["Social Media", "unreachable_states"] == "Direct Traffic; Display Ads"
    assert t.loc["Search Ads", "unreachable_states"] == ""


def test_states_cut_off_from_conversion_are_reported(example):
    P, counts = example
    t = removal_effects(P, counts=counts).set_index("channel")
    # Direct Traffic's only exit is Email, so removing Email strands it.
    assert t.loc["Email", "cut_off_states"] == "Direct Traffic"
    # ... and its probability of converting really is zero without Email.
    assert absorption_probabilities(remove_channel(P, "Email"))["Direct Traffic"] == 0.0
    assert t.loc["Search Ads", "cut_off_states"] == ""


def test_removing_the_only_channel_gives_effect_of_one():
    P, _ = matrices(["Start > A > Conversion", "Start > A > Null"])
    t = removal_effects(P).set_index("channel")
    assert t.loc["A", "conversion_probability_without_channel"] == 0.0
    assert t.loc["A", "removal_effect"] == 1.0


def test_unreachable_channel_at_baseline_is_flagged():
    P, _ = matrices(["Start > A > Conversion", "Start > A > Null"])
    P = P.copy()  # hand-build a matrix where B exists but nothing leads to it
    P.loc["B"] = 0.0
    P["B"] = 0.0
    P.loc["B", CONVERSION] = 1.0
    P = P.loc[[START, "A", "B", CONVERSION, NULL], [START, "A", "B", CONVERSION, NULL]]
    t = removal_effects(P).set_index("channel")
    assert t.loc["B", "status"] == "not_reachable_at_baseline"
    assert t.loc["B", "removal_effect"] == 0.0


def test_unknown_channel_to_remove_and_duplicates_raise(example):
    P, _ = example
    with pytest.raises(ValueError, match="not a state"):
        remove_channel(P, "Ghost")
    with pytest.raises(ValueError, match="duplicate"):
        removal_effects(P, channels=["Email", "Email"])


def test_inconsistent_probabilities_raise_instead_of_returning_negative_effects():
    with pytest.raises(ValueError, match="cannot increase"):
        removal_effect_from_probabilities(0.5, 0.7)
    with pytest.raises(ValueError):
        removal_effect_from_probabilities(float("nan"), 0.1)


# ---------------------------------------------------------------- segments and export
def two_segment_journeys():
    rows = (
        [("new", "Start > Email > Conversion", 1)] * 3
        + [("new", "Start > Email > Null", 0)]
        + [("returning", "Start > Social > Conversion", 1)] * 2
    )
    return pd.DataFrame({
        "journey_id": [f"j{i}" for i in range(len(rows))],
        "segment": [r[0] for r in rows], "path": [r[1] for r in rows], "converted": [r[2] for r in rows],
    })


def test_segment_tables_have_hand_calculated_values():
    # all-journeys model: CR = 5/6. Remove Email -> 1/3 -> RE 3/5. Remove Social -> 1/2 -> RE 2/5.
    table = all_removal_effects(fit_segment_models(two_segment_journeys(), min_journeys=2, min_conversions=1))
    assert table["segment"].tolist() == ["all", "all", "new", "returning"]
    a = table[table["segment"] == "all"].set_index("channel")
    assert a.loc["Email", "removal_effect"] == pytest.approx(3 / 5)
    assert a.loc["Social", "removal_effect"] == pytest.approx(2 / 5)
    # inside a segment that only saw one channel, removing it removes everything
    assert table[(table.segment == "new")].iloc[0]["removal_effect"] == pytest.approx(1.0)
    assert table[(table.segment == "returning")].iloc[0]["removal_effect"] == pytest.approx(1.0)


def test_export_round_trip(tmp_path, example):
    P, counts = example
    table = removal_effects(P, counts=counts, segment="demo")
    out = export_markov_results(table, tmp_path / "nested" / "markov_results.csv")
    back = pd.read_csv(out)
    assert list(back.columns) == RESULT_COLUMNS
    assert len(back) == len(table)
    numeric = back[["baseline_conversion_probability", "conversion_probability_without_channel", "removal_effect"]]
    assert np.isfinite(numeric.to_numpy()).all()
    assert back.set_index("channel").loc["Email", "removal_effect"] == pytest.approx(2 / 7)


# ---------------------------------------------------------------- the real file
REAL_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "multi_touch_attribution_data.csv"
needs_real = pytest.mark.skipif(not REAL_FILE.exists(), reason="real Kaggle file not present in data/raw/")


@pytest.fixture(scope="module")
def real_journeys():
    return build_journeys(clean_events(REAL_FILE, KAGGLE_CONFIG).clean).journeys


@needs_real
def test_real_data_results_are_finite_bounded_and_consistent(real_journeys):
    ms = fit_segment_models(real_journeys)
    table = all_removal_effects(ms)
    assert len(table) == 18 and set(table["segment"]) == {"all", "new", "returning"}
    assert (table.groupby("segment").size() == 6).all()
    assert (table["status"] == "ok").all()
    assert table["removal_effect"].between(0, 1).all()
    assert (table["conversion_probability_without_channel"] < table["baseline_conversion_probability"]).all()
    for seg, model in ms.models.items():
        sub = table[table["segment"] == seg]
        assert sub["baseline_conversion_probability"].iloc[0] == pytest.approx(model.base_conversion_probability)
        # n_visits = arrivals into each channel = the channel columns of the count matrix
        assert sub["n_visits"].sum() == int(model.counts[eligible_channels(model.probs)].to_numpy().sum())


@needs_real
def test_real_data_removal_matches_independent_computation(real_journeys):
    # Recompute one channel's CR_without from scratch with the explicit redirect formula.
    m = fit_markov_model(real_journeys)
    P = m.probs.copy()
    ch = "Referral"
    P[NULL] = P[NULL] + P[ch]
    P[ch] = 0.0
    P.loc[ch] = 0.0
    P.loc[ch, NULL] = 1.0
    expected = conversion_probability(P)
    got = removal_effects(m.probs).set_index("channel").loc[ch, "conversion_probability_without_channel"]
    assert got == pytest.approx(expected)
    assert reachable_states(m.probs) >= set(eligible_channels(m.probs))  # every channel reachable at baseline
