"""Stage 4 tests.

EXAMPLE_PATHS are 9 REAL journeys from the Kaggle file (the ones we traced by hand in Stage 3).
Every other path list is a small TEST FIXTURE, invented only to isolate one rule. Fixtures never
feed any result. The last tests run on the real file.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mta.data_pipeline import KAGGLE_CONFIG, clean_events
from mta.journey_builder import build_journeys, export_journeys
from mta.markov_engine import (
    CONVERSION,
    NULL,
    START,
    absorption_probabilities,
    build_count_matrix,
    build_probability_matrix,
    conversion_probability,
    conversion_probability_iterative,
    count_transitions,
    fit_markov_model,
    fit_segment_models,
    load_journeys,
    matrix_problems,
    parse_path,
    state_order,
    transition_table,
    validate_probability_matrix,
)

EXAMPLE_PATHS = [
    "Start > Social Media > Direct Traffic > Email > Conversion",       # 10062#1
    "Start > Social Media > Display Ads > Display Ads > Conversion",    # 10167#1
    "Start > Search Ads > Conversion",                                  # 10068#1
    "Start > Social Media > Social Media > Conversion",                 # 10068#2
    "Start > Search Ads > Conversion",                                  # 10068#3
    "Start > Social Media > Null",                                      # 10068#4
    "Start > Email > Conversion",                                       # 21704#1
    "Start > Search Ads > Conversion",                                  # 21704#2
    "Start > Social Media > Null",                                      # 21704#3
]


def states_of(paths):
    return [parse_path(p) for p in paths]


def probs_of(paths):
    return build_probability_matrix(build_count_matrix(count_transitions(states_of(paths))))


# ---------------------------------------------------------------- parsing
def test_parse_path_splits_into_states():
    assert parse_path("Start > A > A > Conversion") == ["Start", "A", "A", "Conversion"]


@pytest.mark.parametrize("bad", [
    "A > B > Conversion",                  # no Start
    "Start > A > B",                       # no terminal state
    "Start > Conversion > A > Null",       # terminal state inside the path
    "Start > A > Start > Null",            # Start inside the path
    "Start",                               # too short
    "",                                    # empty
    "Start >  > Null",                     # empty state name
])
def test_invalid_paths_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_path(bad)


# ---------------------------------------------------------------- counts (hand-calculated)
def test_transition_counts_match_hand_calculation():
    counts = count_transitions(states_of(EXAMPLE_PATHS))
    assert sum(counts.values()) == 23  # path lengths: 4+4+2+3+2+2+2+2+2 transitions
    expected = {
        (START, "Social Media"): 5, (START, "Search Ads"): 3, (START, "Email"): 1,
        ("Social Media", "Direct Traffic"): 1, ("Social Media", "Display Ads"): 1,
        ("Social Media", "Social Media"): 1, ("Social Media", CONVERSION): 1, ("Social Media", NULL): 2,
        ("Direct Traffic", "Email"): 1,
        ("Email", CONVERSION): 2,
        ("Display Ads", "Display Ads"): 1, ("Display Ads", CONVERSION): 1,
        ("Search Ads", CONVERSION): 3,
    }
    assert dict(counts) == expected


def test_repeated_channel_is_a_self_loop_transition():
    counts = count_transitions(states_of(["Start > A > A > A > Null"]))
    assert counts[("A", "A")] == 2
    assert counts[(START, "A")] == 1 and counts[("A", NULL)] == 1


def test_state_order_is_deterministic():
    counts = count_transitions(states_of(["Start > B > A > Null", "Start > C > Conversion"]))
    assert state_order(counts) == [START, "A", "B", "C", CONVERSION, NULL]


def test_count_matrix_layout_and_totals():
    cm = build_count_matrix(count_transitions(states_of(EXAMPLE_PATHS)))
    assert cm.shape == (8, 8)
    assert cm.loc[START, "Social Media"] == 5
    assert cm.to_numpy().sum() == 23
    assert cm.loc[CONVERSION].sum() == 0 and cm.loc[NULL].sum() == 0  # nothing leaves terminals
    assert cm[START].sum() == 0  # nothing enters Start


def test_transition_table_probabilities():
    t = transition_table(count_transitions(states_of(EXAMPLE_PATHS)))
    row = t[(t.from_state == "Social Media") & (t.to_state == NULL)].iloc[0]
    assert row["count"] == 2 and row["from_total"] == 6 and row["probability"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- probabilities
def test_probability_matrix_matches_hand_calculation():
    P = probs_of(EXAMPLE_PATHS)
    assert P.loc[START, "Social Media"] == pytest.approx(5 / 9)
    assert P.loc[START, "Search Ads"] == pytest.approx(3 / 9)
    assert P.loc[START, "Email"] == pytest.approx(1 / 9)
    for to in ("Direct Traffic", "Display Ads", "Social Media", CONVERSION):
        assert P.loc["Social Media", to] == pytest.approx(1 / 6)
    assert P.loc["Social Media", NULL] == pytest.approx(2 / 6)
    assert P.loc["Display Ads", "Display Ads"] == pytest.approx(0.5)
    assert P.loc["Display Ads", CONVERSION] == pytest.approx(0.5)
    assert P.loc["Email", CONVERSION] == 1.0 and P.loc["Search Ads", CONVERSION] == 1.0


def test_every_row_sums_to_one():
    P = probs_of(EXAMPLE_PATHS)
    assert np.allclose(P.sum(axis=1).to_numpy(), 1.0)


def test_terminal_states_are_absorbing():
    P = probs_of(EXAMPLE_PATHS)
    assert P.loc[CONVERSION, CONVERSION] == 1.0 and P.loc[NULL, NULL] == 1.0
    assert P.loc[CONVERSION].drop(CONVERSION).sum() == 0 and P.loc[NULL].drop(NULL).sum() == 0
    # Stepping the chain 50 times never lets terminal probability mass leak out.
    Pk = np.linalg.matrix_power(P.to_numpy(), 50)
    i_c, i_n = list(P.index).index(CONVERSION), list(P.index).index(NULL)
    assert Pk[i_c, i_c] == pytest.approx(1.0) and Pk[i_n, i_n] == pytest.approx(1.0)


def test_nothing_transitions_into_start():
    assert probs_of(EXAMPLE_PATHS)[START].sum() == 0


def test_non_terminal_row_with_no_departures_raises():
    cm = pd.DataFrame(0, index=[START, "A", CONVERSION, NULL], columns=[START, "A", CONVERSION, NULL])
    cm.loc[START, "A"] = 3  # A is reachable but nothing leaves it
    with pytest.raises(ValueError, match="no outgoing transitions"):
        build_probability_matrix(cm)


def test_counts_leaving_a_terminal_state_are_rejected():
    counts = {(START, "A"): 1, ("A", CONVERSION): 1, (CONVERSION, "A"): 1}
    with pytest.raises(ValueError, match="terminal"):
        build_count_matrix(counts)


def test_validator_catches_corrupted_matrices():
    P = probs_of(EXAMPLE_PATHS)
    assert matrix_problems(P) == []
    bad_sum = P.copy()
    bad_sum.loc["Email", CONVERSION] = 0.9
    assert any("sums to" in p for p in matrix_problems(bad_sum))
    leaky = P.copy()
    leaky.loc[CONVERSION, CONVERSION] = 0.5
    leaky.loc[CONVERSION, NULL] = 0.5
    assert any("not absorbing" in p for p in matrix_problems(leaky))
    into_start = P.copy()
    into_start.loc["Email", START] = 0.2
    into_start.loc["Email", CONVERSION] = 0.8
    assert any("into Start" in p for p in matrix_problems(into_start))
    with pytest.raises(ValueError):
        validate_probability_matrix(bad_sum)


# ---------------------------------------------------------------- eventual conversion probability
def test_eventual_conversion_probability_matches_hand_calculation():
    # h(Social) = 1/6*1 (Direct) + 1/6*1 (Display) + 1/6*h(Social) + 1/6*1 (Conversion) + 2/6*0
    #   => (5/6) h = 1/2 => h = 3/5.   Start = 5/9*3/5 + 3/9*1 + 1/9*1 = 7/9.
    P = probs_of(EXAMPLE_PATHS)
    h = absorption_probabilities(P)
    assert h["Social Media"] == pytest.approx(0.6)
    for s in ("Direct Traffic", "Display Ads", "Email", "Search Ads"):
        assert h[s] == pytest.approx(1.0)
    assert conversion_probability(P) == pytest.approx(7 / 9)


def test_eventual_probability_differs_from_direct_start_to_conversion():
    P = probs_of(EXAMPLE_PATHS)
    assert P.loc[START, CONVERSION] == 0.0          # nobody converts with zero touches
    assert conversion_probability(P) == pytest.approx(7 / 9)  # but most journeys get there eventually


def test_direct_and_indirect_routes_both_count():
    # Fixture: Start -> Conversion directly (1/4) PLUS Start -> A -> Conversion (1/2 * 1/2) = 1/2.
    paths = ["Start > Conversion", "Start > Null", "Start > A > Conversion", "Start > A > Null"]
    P = probs_of(paths)
    assert P.loc[START, CONVERSION] == pytest.approx(0.25)
    assert conversion_probability(P) == pytest.approx(0.5)


def test_loops_are_summed_correctly():
    # Fixture: A -> A with prob 1/2, A -> Conversion 1/2. Geometric series => eventual prob 1.
    P = probs_of(["Start > A > A > Conversion", "Start > A > Conversion"])
    assert P.loc["A", "A"] == pytest.approx(1 / 3)
    assert absorption_probabilities(P)["A"] == pytest.approx(1.0)


def test_iterative_stepping_agrees_with_linear_solve():
    P = probs_of(EXAMPLE_PATHS)
    assert conversion_probability_iterative(P) == pytest.approx(conversion_probability(P), abs=1e-9)


def test_closed_loop_without_exit_raises():
    cm = pd.DataFrame(0, index=[START, "X", CONVERSION, NULL], columns=[START, "X", CONVERSION, NULL])
    cm.loc[START, "X"] = 1
    cm.loc["X", "X"] = 1  # X can never leave
    P = build_probability_matrix(cm)
    with pytest.raises(ValueError, match="closed loop"):
        conversion_probability(P)


def test_model_reproduces_observed_conversion_rate():
    # Flow conservation: arrivals == departures for every channel, so the fitted chain's
    # absorption probability equals converting journeys / all journeys.
    journeys = pd.DataFrame({
        "journey_id": [f"j{i}" for i in range(len(EXAMPLE_PATHS))],
        "path": EXAMPLE_PATHS,
        "segment": ["new"] * len(EXAMPLE_PATHS),
        "converted": [int(p.endswith("Conversion")) for p in EXAMPLE_PATHS],
    })
    m = fit_markov_model(journeys)
    assert m.base_conversion_probability == pytest.approx(7 / 9)
    assert m.observed_conversion_rate == pytest.approx(7 / 9)
    assert m.reproduces_observed_rate


# ---------------------------------------------------------------- loading and consistency
def test_load_journeys_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_stage3"):
        load_journeys(tmp_path / "nope.csv")


def test_load_journeys_rejects_missing_columns(tmp_path):
    p = tmp_path / "j.csv"
    p.write_text("journey_id,path\nj1,Start > A > Null\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_journeys(p)


def test_converted_flag_must_agree_with_path():
    j = pd.DataFrame({"journey_id": ["j1"], "path": ["Start > A > Conversion"],
                      "segment": ["new"], "converted": [0]})
    with pytest.raises(ValueError, match="disagrees"):
        fit_markov_model(j)


def test_export_then_load_round_trip(tmp_path):
    j = pd.DataFrame({"journey_id": ["j1", "j2"], "user_id": ["u1", "u2"],
                      "path": ["Start > A > Conversion", "Start > A > Null"],
                      "segment": ["new", "new"], "converted": [1, 0]})
    out = export_journeys(j, tmp_path / "journeys.csv")
    loaded = load_journeys(out)
    assert fit_markov_model(loaded).base_conversion_probability == pytest.approx(0.5)


# ---------------------------------------------------------------- segments
def segment_fixture():
    rows = (
        [("new", "Start > Email > Conversion", 1)] * 3
        + [("new", "Start > Email > Null", 0)]
        + [("returning", "Start > Social > Conversion", 1)] * 2
    )
    return pd.DataFrame({
        "journey_id": [f"j{i}" for i in range(len(rows))],
        "segment": [r[0] for r in rows], "path": [r[1] for r in rows], "converted": [r[2] for r in rows],
    })


def test_segment_models_are_separate_and_not_mixed():
    ms = fit_segment_models(segment_fixture(), min_journeys=2, min_conversions=1)
    assert set(ms.models) == {"all", "new", "returning"}
    assert "Social" not in ms.models["new"].probs.index      # new customers never saw Social
    assert "Email" not in ms.models["returning"].probs.index
    assert ms.models["new"].base_conversion_probability == pytest.approx(0.75)
    assert ms.models["returning"].base_conversion_probability == pytest.approx(1.0)
    assert ms.models["all"].n_journeys == 6


def test_small_segments_are_skipped_with_a_reason():
    ms = fit_segment_models(segment_fixture(), min_journeys=3, min_conversions=1)
    assert "returning" not in ms.models
    assert "only 2 journeys" in ms.skipped["returning"]
    ms2 = fit_segment_models(segment_fixture(), min_journeys=1, min_conversions=3)
    assert "returning" in ms2.skipped and "converting" in ms2.skipped["returning"]


def test_unknown_segment_value_raises():
    j = segment_fixture()
    j.loc[0, "segment"] = "vip"
    with pytest.raises(ValueError, match="unknown segment"):
        fit_segment_models(j)


def test_fit_for_unknown_segment_name_raises():
    with pytest.raises(ValueError):
        fit_markov_model(segment_fixture(), segment="vip")


# ---------------------------------------------------------------- the real file
REAL_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "multi_touch_attribution_data.csv"
needs_real = pytest.mark.skipif(not REAL_FILE.exists(), reason="real Kaggle file not present in data/raw/")


@pytest.fixture(scope="module")
def real_journeys():
    return build_journeys(clean_events(REAL_FILE, KAGGLE_CONFIG).clean).journeys


@needs_real
def test_real_data_matrices_are_valid_and_complete(real_journeys):
    m = fit_markov_model(real_journeys)
    assert matrix_problems(m.probs) == []
    expected_transitions = sum(len(p.split(" > ")) - 1 for p in real_journeys["path"])
    assert int(m.counts.to_numpy().sum()) == expected_transitions  # every transition counted once
    assert m.direct_start_to_conversion == 0.0
    assert m.n_journeys == 6382 and m.n_converting == 4944
    assert m.base_conversion_probability == pytest.approx(4944 / 6382)
    assert m.reproduces_observed_rate


@needs_real
def test_real_data_segment_models(real_journeys):
    ms = fit_segment_models(real_journeys)
    assert set(ms.models) == {"all", "new", "returning"} and ms.skipped == {}
    assert ms.models["new"].n_journeys == 2847 and ms.models["returning"].n_journeys == 3535
    assert ms.models["new"].base_conversion_probability == pytest.approx(2381 / 2847)
    assert ms.models["returning"].base_conversion_probability == pytest.approx(2563 / 3535)
    for model in ms.models.values():
        assert matrix_problems(model.probs) == []
