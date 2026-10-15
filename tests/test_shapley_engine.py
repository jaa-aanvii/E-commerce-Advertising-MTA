"""Stage 8 / Task 1.3 tests: Shapley values from coalition values.

All games below are small TEST FIXTURES with answers worked out by hand (shown in comments). They
never feed any project result. The last test runs on the real files and is skipped if missing.
"""
import inspect
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import shapley_engine.shapley_engine as se
from shapley_engine.coalition_generator import coalitions_to_frame, generate_coalitions, save_coalitions
from shapley_engine.coalition_value import calculate_all_coalition_values, save_coalition_values
from shapley_engine.shapley_engine import (
    ShapleyError,
    build_and_save,
    build_value_lookup,
    compute_shapley_values,
    load_shapley_values,
    save_shapley_values,
    shapley_to_frame,
    shapley_value,
    shapley_weight,
    validate_shapley_values,
)
from mta.journey_builder import DEFAULT_OUTPUT_PATH
from mta.markov_engine import fit_markov_model

TOL = 1e-12


def game(players, fn):
    """Lookup {frozenset: V} for the full power set of `players`, V given by fn(frozenset)."""
    return {frozenset(c): float(fn(frozenset(c))) for k in range(len(players) + 1)
            for c in combinations(sorted(players), k)}


def frame_of(lookup):
    """A coalition-values-like DataFrame from a lookup, in canonical coalition order."""
    table = coalitions_to_frame(sorted(lookup, key=lambda c: (len(c), tuple(sorted(c)))))
    table["value"] = [lookup[frozenset(s.split("|")) if s else frozenset()] for s in table["coalition"]]
    return table


# ---------------------------------------------------------------- weights
def test_weights_for_three_players_by_hand():
    assert shapley_weight(3, 0) == pytest.approx(1 / 3, abs=TOL)  # 0! 2! / 3! = 2/6
    assert shapley_weight(3, 1) == pytest.approx(1 / 6, abs=TOL)  # 1! 1! / 3! = 1/6
    assert shapley_weight(3, 2) == pytest.approx(1 / 3, abs=TOL)  # 2! 0! / 3! = 2/6


def test_weights_over_all_coalitions_not_containing_a_channel_sum_to_one():
    for n in range(1, 9):
        assert sum(comb(n - 1, s) * shapley_weight(n, s) for s in range(n)) == pytest.approx(1.0, abs=1e-12)


def test_weight_rejects_impossible_sizes():
    for n, s in [(0, 0), (3, 3), (3, -1)]:
        with pytest.raises(ValueError):
            shapley_weight(n, s)


# ---------------------------------------------------------------- hand-calculated games
def test_two_player_game_by_hand():
    # V(empty)=0, V(A)=1, V(B)=0, V(AB)=2
    # phi_A = 1/2*(V(A)-V(0)) + 1/2*(V(AB)-V(B)) = .5*1 + .5*2 = 1.5
    # phi_B = 1/2*(V(B)-V(0)) + 1/2*(V(AB)-V(A)) = .5*0 + .5*1 = 0.5
    v = {frozenset(): 0, frozenset("A"): 1, frozenset("B"): 0, frozenset("AB"): 2}
    phi = compute_shapley_values(v)
    assert phi == {"A": pytest.approx(1.5, abs=TOL), "B": pytest.approx(0.5, abs=TOL)}


def test_three_player_game_by_hand():
    # V(AB)=V(AC)=V(ABC)=1, everything else 0. Over the 6 arrival orders A is pivotal in 4:
    # phi_A = 4/6 = 2/3, phi_B = 1/6, phi_C = 1/6 (sum 1 = V(N) - V(0)).
    wins = {frozenset("AB"), frozenset("AC"), frozenset("ABC")}
    phi = compute_shapley_values(game("ABC", lambda s: 1 if s in wins else 0))
    assert phi["A"] == pytest.approx(2 / 3, abs=TOL)
    assert phi["B"] == pytest.approx(1 / 6, abs=TOL)
    assert phi["C"] == pytest.approx(1 / 6, abs=TOL)


def test_dictator_game_gives_everything_to_one_player_and_nothing_to_null_players():
    phi = compute_shapley_values(game("ABC", lambda s: 1 if "A" in s else 0))
    assert phi["A"] == pytest.approx(1.0, abs=TOL)
    assert phi["B"] == pytest.approx(0.0, abs=TOL) and phi["C"] == pytest.approx(0.0, abs=TOL)


def test_nonzero_empty_coalition_is_excluded_from_the_split():
    # V(0)=.25, V(A)=.25, V(B)=.5, V(AB)=.75  (this is the Task 1.2 chain example)
    # phi_A = .5*(.25-.25) + .5*(.75-.5) = .125 ; phi_B = .5*(.5-.25) + .5*(.75-.25) = .375
    v = {frozenset(): .25, frozenset("A"): .25, frozenset("B"): .5, frozenset("AB"): .75}
    phi = compute_shapley_values(v)
    assert phi["A"] == pytest.approx(0.125, abs=TOL) and phi["B"] == pytest.approx(0.375, abs=TOL)
    assert sum(phi.values()) == pytest.approx(0.5, abs=TOL)  # V(N) - V(0), not V(N)


def test_one_player_and_zero_players():
    assert compute_shapley_values({frozenset(): 0.2, frozenset({"A"}): 0.7})["A"] == pytest.approx(0.5, abs=TOL)
    assert compute_shapley_values({frozenset(): 0.3}) == {}


# ---------------------------------------------------------------- symmetry
def test_symmetric_game_gives_equal_values():
    # V depends only on |S|: every channel is interchangeable, so phi_i = (V(N) - V(0)) / n.
    sizes = {0: 0.1, 1: 0.2, 2: 0.55, 3: 0.6, 4: 0.9}
    phi = compute_shapley_values(game("ABCD", lambda s: sizes[len(s)]))
    for c in "ABCD":
        assert phi[c] == pytest.approx((0.9 - 0.1) / 4, abs=TOL)


def test_two_interchangeable_channels_in_an_asymmetric_game_get_equal_values():
    # B and C are symmetric (same effect everywhere); A is different.
    def v(s):
        return 0.5 * ("A" in s) + 0.3 * len(s & {"B", "C"}) ** 2
    phi = compute_shapley_values(game("ABC", v))
    assert phi["B"] == pytest.approx(phi["C"], abs=TOL)
    assert phi["A"] != pytest.approx(phi["B"], abs=1e-3)


# ---------------------------------------------------------------- efficiency and general properties
def test_efficiency_holds_on_arbitrary_games():
    rng = np.random.RandomState(7)
    for n in range(1, 7):
        players = [f"c{i}" for i in range(n)]
        v = game(players, lambda s: rng.uniform(-1, 1))
        phi = compute_shapley_values(v)
        info = validate_shapley_values(phi, v)
        assert info["sum_shapley"] == pytest.approx(v[frozenset(players)] - v[frozenset()], abs=1e-12)


def test_additive_game_gives_each_player_its_own_value():
    worth = {"A": 0.1, "B": 0.25, "C": 0.05}
    phi = compute_shapley_values(game("ABC", lambda s: sum(worth[c] for c in s)))
    for c, w in worth.items():
        assert phi[c] == pytest.approx(w, abs=TOL)


def test_shapley_value_single_channel_matches_full_computation():
    v = game("ABC", lambda s: len(s) ** 2 / 9)
    assert shapley_value(v, "B") == compute_shapley_values(v)["B"]
    with pytest.raises(ShapleyError):
        shapley_value(v, "Z")


# ---------------------------------------------------------------- determinism
def test_deterministic_and_independent_of_row_order():
    v = game("ABCD", lambda s: (len(s) ** 2 + ("A" in s) * 0.3) / 20)
    shuffled = dict(reversed(list(v.items())))
    a, b, c = compute_shapley_values(v), compute_shapley_values(v), compute_shapley_values(shuffled)
    assert list(a.items()) == list(b.items()) == list(c.items())  # identical values AND order


def test_input_lookup_is_not_modified():
    v = game("ABC", lambda s: len(s) / 3)
    before = dict(v)
    compute_shapley_values(v)
    assert v == before


def test_csv_output_is_byte_identical_across_runs(tmp_path):
    phi = compute_shapley_values(game("ABC", lambda s: len(s) ** 2 / 9))
    a = save_shapley_values(phi, tmp_path / "a.csv")
    b = save_shapley_values(dict(reversed(list(phi.items()))), tmp_path / "b.csv")
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------- missing / invalid values
def test_missing_coalition_value_is_detected():
    v = game("ABC", lambda s: len(s) / 3)
    del v[frozenset("AB")]
    with pytest.raises(ShapleyError, match="missing coalition value V\\({A, B}\\)"):
        compute_shapley_values(v)


def test_missing_empty_or_grand_coalition_is_detected():
    v = game("ABC", lambda s: len(s) / 3)
    no_empty = {k: x for k, x in v.items() if k}
    with pytest.raises(ShapleyError, match="empty"):
        compute_shapley_values(no_empty)
    no_grand = {k: x for k, x in v.items() if k != frozenset("ABC")}
    with pytest.raises(ShapleyError, match="missing coalition value"):
        compute_shapley_values(no_grand)


def test_lookup_rejects_nan_inf_duplicates_and_non_numeric():
    base = frame_of(game("AB", lambda s: len(s) / 2))
    for bad in (np.nan, np.inf, -np.inf):
        t = base.copy(); t.loc[1, "value"] = bad
        with pytest.raises(ShapleyError, match="not finite"):
            build_value_lookup(t)
    with pytest.raises(ShapleyError, match="duplicate"):
        build_value_lookup(pd.concat([base, base.iloc[[1]]], ignore_index=True))
    t = base.copy(); t["value"] = t["value"].astype(str)
    with pytest.raises(ShapleyError, match="numeric"):
        build_value_lookup(t)
    with pytest.raises(ShapleyError, match="'value' column"):
        build_value_lookup(base.drop(columns="value"))


def test_validation_detects_wrong_shapley_results():
    v = game("ABC", lambda s: len(s) / 3)
    good = compute_shapley_values(v)
    validate_shapley_values(good, v)
    with pytest.raises(ShapleyError, match="efficiency"):
        validate_shapley_values({**good, "A": good["A"] + 0.01}, v)
    with pytest.raises(ShapleyError, match="exactly one"):
        validate_shapley_values({"A": good["A"], "B": good["B"]}, v)
    with pytest.raises(ShapleyError, match="exactly one"):
        validate_shapley_values({**good, "Z": 0.0}, v)
    with pytest.raises(ShapleyError, match="finite"):
        validate_shapley_values({**good, "A": float("nan")}, v)
    with pytest.raises(ShapleyError, match="finite"):
        validate_shapley_values({**good, "A": float("inf")}, v)


# ---------------------------------------------------------------- integration with Tasks 1.1 / 1.2
def journeys_frame(paths):
    return pd.DataFrame({"journey_id": [f"u{i}#1" for i in range(len(paths))], "path": paths,
                         "segment": "new", "converted": [int(p.endswith("Conversion")) for p in paths]})


# Task 1.2 chain fixture: V(0)=.25, V(A)=.25, V(B)=.5, V(AB)=.75  ->  phi_A=.125, phi_B=.375
THREE = ["Start > A > B > Conversion", "Start > A > Null", "Start > B > Conversion", "Start > Conversion"]


def test_end_to_end_from_task_1_2_values(tmp_path):
    P = fit_markov_model(journeys_frame(THREE)).probs
    values = calculate_all_coalition_values(P, coalitions_to_frame(generate_coalitions(["A", "B"])))
    vp = save_coalition_values(values, tmp_path / "coalition_values.csv")
    r = build_and_save(vp, tmp_path / "shapley_values.csv")
    assert r["shapley"]["A"] == pytest.approx(0.125, abs=1e-9)
    assert r["shapley"]["B"] == pytest.approx(0.375, abs=1e-9)
    assert r["sum_shapley"] == pytest.approx(r["v_grand"] - r["v_empty"], abs=1e-9)
    assert r["v_grand"] == pytest.approx(0.75, abs=1e-9) and r["v_empty"] == pytest.approx(0.25, abs=1e-9)
    out = pd.read_csv(r["output_path"])
    assert list(out.columns) == ["channel", "shapley_value"] and out["channel"].tolist() == ["A", "B"]
    assert load_shapley_values(r["output_path"]) == r["shapley"]  # saved at full precision


def test_end_to_end_with_channel_names_containing_spaces(tmp_path):
    paths = ["Start > Search Ads > Social Media > Conversion", "Start > Social Media > Null",
             "Start > Search Ads > Null", "Start > Conversion"]
    P = fit_markov_model(journeys_frame(paths)).probs
    values = calculate_all_coalition_values(P, coalitions_to_frame(generate_coalitions(["Search Ads", "Social Media"])))
    r = build_and_save(save_coalition_values(values, tmp_path / "v.csv"), tmp_path / "s.csv")
    assert sorted(r["shapley"]) == ["Search Ads", "Social Media"]
    assert r["sum_shapley"] == pytest.approx(r["v_grand"] - r["v_empty"], abs=1e-9)


def test_build_and_save_detects_a_values_file_with_a_missing_coalition(tmp_path):
    P = fit_markov_model(journeys_frame(THREE)).probs
    values = calculate_all_coalition_values(P, coalitions_to_frame(generate_coalitions(["A", "B"])))
    vp = save_coalition_values(values.drop(index=2), tmp_path / "v.csv")
    with pytest.raises(ShapleyError, match="missing coalition value|expected 2\\^"):
        build_and_save(vp, tmp_path / "s.csv")
    assert not (tmp_path / "s.csv").exists()  # nothing written when the input is bad


def test_build_and_save_rejects_grand_value_that_disagrees_with_recorded_baseline(tmp_path):
    P = fit_markov_model(journeys_frame(THREE)).probs
    values = calculate_all_coalition_values(P, coalitions_to_frame(generate_coalitions(["A", "B"])))
    values["baseline_conversion_probability"] = 0.9
    with pytest.raises(ShapleyError, match="baseline"):
        build_and_save(save_coalition_values(values, tmp_path / "v.csv"), tmp_path / "s.csv")


def test_missing_input_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_stage7"):
        build_and_save(tmp_path / "nope.csv", tmp_path / "s.csv")


def test_output_has_only_channel_and_shapley_value_no_revenue_or_spend():
    df = shapley_to_frame({"B": 0.2, "A": 0.1})
    assert list(df.columns) == ["channel", "shapley_value"] and df["channel"].tolist() == ["A", "B"]
    names = " ".join(inspect.signature(f).__str__() for f in (se.compute_shapley_values, se.shapley_value, se.build_and_save))
    assert not any(w in names.lower() for w in ("revenue", "spend", "cost", "roi"))


def test_save_and_load_roundtrip_keeps_full_precision(tmp_path):
    phi = compute_shapley_values(game("ABC", lambda s: (len(s) ** 3) / 27 + 1 / 7))
    assert load_shapley_values(save_shapley_values(phi, tmp_path / "x" / "s.csv")) == phi


# ---------------------------------------------------------------- real data
def test_real_data_efficiency(tmp_path):
    vp = Path("data/processed/coalition_values.csv")
    if not vp.exists():
        pytest.skip("coalition_values.csv not found; run scripts/run_stage3.py, run_stage6.py, run_stage7.py first")
    r = build_and_save(vp, tmp_path / "shapley_values.csv")
    assert r["n_coalitions"] == 2 ** r["n_channels"]
    assert abs(r["sum_shapley"] - (r["v_grand"] - r["v_empty"])) <= 1e-9
