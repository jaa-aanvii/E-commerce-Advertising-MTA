"""Tests for Task 1.4 - Shapley validation.

ALL games and tables in this file are SYNTHETIC (hand-made fixtures used to prove
the validator is correct).  None of the numbers here are project results.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shapley_engine.shapley_validation import (  # noqa: E402
    FAIL,
    HAND_GAMES,
    PASS,
    SKIPPED,
    SchemaError,
    all_coalitions,
    parse_coalition,
    parse_coalition_with_issue,
    reference_shapley,
    reference_shapley_by_permutations,
    reference_shapley_by_subsets,
    validate_shapley_files,
    validate_shapley_outputs,
)

F = frozenset

# --- synthetic fixtures (written independently of the validator's own HAND_GAMES) ---
PROJECT_GAME = {F(): 0.0, F("A"): 0.4, F("B"): 0.2, F("AB"): 0.8}  # phi_A=0.5, phi_B=0.3
PROJECT_PHI = {"A": 0.5, "B": 0.3}

GLOVE = {
    F(): 0.0,
    F(["L1"]): 0.0,
    F(["L2"]): 0.0,
    F(["R"]): 0.0,
    F(["L1", "L2"]): 0.0,
    F(["L1", "R"]): 1.0,
    F(["L2", "R"]): 1.0,
    F(["L1", "L2", "R"]): 1.0,
}
GLOVE_PHI = {"L1": 1 / 6, "L2": 1 / 6, "R": 2 / 3}

NEGATIVE_GAME = {F(): 0.0, F("A"): 0.6, F("B"): 0.1, F("AB"): 0.4}  # phi_A=0.45, phi_B=-0.05
NEGATIVE_PHI = {"A": 0.45, "B": -0.05}

NULL_GAME = {F(): 0.0, F("A"): 0.5, F("D"): 0.0, F("AD"): 0.5}  # D adds nothing


def coalition_df(values, sep="|"):
    rows = [
        {"coalition": sep.join(sorted(k)), "value": v}
        for k, v in sorted(values.items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))
    ]
    return pd.DataFrame(rows)


def shapley_df(phi):
    return pd.DataFrame({"channel": list(phi), "shapley_value": list(phi.values())})


def status(report, number):
    return next(r.status for r in report.results if r.number == number)


def statuses(report):
    return {r.number: r.status for r in report.results}


# --------------------------------------------------------------------------- #
# reference calculation + hand-calculated games
# --------------------------------------------------------------------------- #
def test_project_hand_game_matches_hand_calculation():
    phi = reference_shapley(["A", "B"], PROJECT_GAME)
    assert phi["A"] == pytest.approx(0.5)
    assert phi["B"] == pytest.approx(0.3)
    assert sum(phi.values()) == pytest.approx(PROJECT_GAME[F("AB")] - PROJECT_GAME[F()])


@pytest.mark.parametrize("game", HAND_GAMES, ids=lambda g: g.name)
def test_reference_matches_every_hand_game(game):
    phi = reference_shapley(game.players, game.values)
    for player, expected in game.expected.items():
        assert phi[player] == pytest.approx(expected, abs=1e-12)


def test_permutation_and_subset_methods_agree_on_random_game():
    rng = random.Random(7)
    players = ["a", "b", "c", "d", "e"]
    values = {s: rng.random() for s in all_coalitions(players)}
    by_perm = reference_shapley_by_permutations(players, values)
    by_subset = reference_shapley_by_subsets(players, values)
    for p in players:
        assert by_perm[p] == pytest.approx(by_subset[p], abs=1e-12)
    assert sum(by_perm.values()) == pytest.approx(values[F(players)] - values[F()], abs=1e-12)


def test_reference_rejects_incomplete_game():
    with pytest.raises(ValueError):
        reference_shapley(["A", "B"], {F(): 0.0, F("A"): 0.1, F("AB"): 0.5})


# --------------------------------------------------------------------------- #
# coalition parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cell, expected",
    [
        ("", F()),
        ("∅", F()),
        ("{}", F()),
        (float("nan"), F()),
        ("A", F(["A"])),
        ("A|B", F(["A", "B"])),
        ("B, A", F(["A", "B"])),
        ("{Google, Email}", F(["Google", "Email"])),
        ("A;B+C", F(["A", "B", "C"])),
        ("Paid Search|Email", F(["Paid Search", "Email"])),
    ],
)
def test_parse_coalition(cell, expected):
    assert parse_coalition(cell) == expected


@pytest.mark.parametrize("cell", ["A,,B", "A|A", "A|Null", "Start", "Conversion|A"])
def test_malformed_coalitions_are_flagged(cell):
    assert parse_coalition_with_issue(cell)[1] is not None


# --------------------------------------------------------------------------- #
# happy paths
# --------------------------------------------------------------------------- #
def test_all_checks_pass_on_correct_glove_tables():
    report = validate_shapley_outputs(coalition_df(GLOVE), shapley_df(GLOVE_PHI))
    assert report.passed, report.format()
    assert set(statuses(report).values()) == {PASS}
    assert len(report.results) == 13


def test_project_hand_game_passes_through_csv_files(tmp_path):
    coalition_df(PROJECT_GAME).to_csv(tmp_path / "cv.csv", index=False)  # empty coalition -> blank cell
    shapley_df(PROJECT_PHI).to_csv(tmp_path / "sv.csv", index=False)
    report = validate_shapley_files(tmp_path / "cv.csv", tmp_path / "sv.csv", baseline=0.8)
    assert report.passed, report.format()


def test_negative_shapley_value_is_allowed():
    report = validate_shapley_outputs(coalition_df(NEGATIVE_GAME), shapley_df(NEGATIVE_PHI))
    assert min(NEGATIVE_PHI.values()) < 0
    assert report.passed, report.format()


def test_report_is_deterministic_and_row_order_independent():
    first = validate_shapley_outputs(coalition_df(GLOVE), shapley_df(GLOVE_PHI))
    second = validate_shapley_outputs(coalition_df(GLOVE), shapley_df(GLOVE_PHI))
    shuffled = coalition_df(GLOVE).sample(frac=1, random_state=3).reset_index(drop=True)
    third = validate_shapley_outputs(shuffled, shapley_df(dict(reversed(list(GLOVE_PHI.items())))))
    assert first.format() == second.format()
    assert third.passed, third.format()


# --------------------------------------------------------------------------- #
# coalition table failures (checks 1 and 2)
# --------------------------------------------------------------------------- #
def test_missing_coalition_detected_and_dependents_skipped():
    df = coalition_df(GLOVE)
    df = df[df["coalition"] != "L1|R"]
    report = validate_shapley_outputs(df, shapley_df(GLOVE_PHI))
    st = statuses(report)
    assert st[1] == FAIL
    assert all(st[n] == SKIPPED for n in (9, 10, 12, 13))
    assert "missing" in next(r.detail for r in report.results if r.number == 1)
    assert not report.passed


def test_duplicate_coalition_detected():
    df = pd.concat([coalition_df(GLOVE), coalition_df(GLOVE).iloc[[3]]], ignore_index=True)
    report = validate_shapley_outputs(df, shapley_df(GLOVE_PHI))
    assert status(report, 1) == FAIL
    assert "duplicate" in next(r.detail for r in report.results if r.number == 1)


def test_incomplete_coalition_space_detected():
    df = coalition_df(GLOVE)
    df = df[df["coalition"].str.count(r"\|") == 0]  # only singletons and the empty set
    report = validate_shapley_outputs(df, shapley_df(GLOVE_PHI), expected_channels=["L1", "L2", "R"])
    assert status(report, 1) == FAIL


def test_expected_channels_reveal_a_channel_absent_from_the_table():
    report = validate_shapley_outputs(coalition_df(PROJECT_GAME), shapley_df(PROJECT_PHI), expected_channels=["A", "B", "C"])
    assert status(report, 1) == FAIL


def test_malformed_coalition_row_detected():
    df = coalition_df(PROJECT_GAME)
    df.loc[df["coalition"] == "A", "coalition"] = "A,,"
    assert status(validate_shapley_outputs(df, shapley_df(PROJECT_PHI)), 1) == FAIL


def test_reserved_state_as_channel_detected():
    df = coalition_df(PROJECT_GAME)
    df.loc[df["coalition"] == "B", "coalition"] = "B|Null"
    report = validate_shapley_outputs(df, shapley_df(PROJECT_PHI))
    assert status(report, 1) == FAIL


@pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf"), float("-inf")])
def test_invalid_coalition_value_detected(bad):
    df = coalition_df(PROJECT_GAME)
    df["value"] = df["value"].astype(object)
    df.loc[df["coalition"] == "A", "value"] = bad
    report = validate_shapley_outputs(df, shapley_df(PROJECT_PHI))
    assert status(report, 2) == FAIL
    assert status(report, 9) == SKIPPED


def test_coalition_value_outside_probability_range_detected():
    df = coalition_df(PROJECT_GAME)
    df.loc[df["coalition"] == "A|B", "value"] = 1.5
    assert status(validate_shapley_outputs(df, shapley_df(PROJECT_PHI)), 2) == FAIL


def test_baseline_mismatch_and_match():
    df, sv = coalition_df(PROJECT_GAME), shapley_df(PROJECT_PHI)
    assert status(validate_shapley_outputs(df, sv, baseline=0.8), 2) == PASS
    assert status(validate_shapley_outputs(df, sv, baseline=0.7), 2) == FAIL


# --------------------------------------------------------------------------- #
# Shapley table failures (checks 3-8)
# --------------------------------------------------------------------------- #
def test_missing_channel_detected():
    report = validate_shapley_outputs(coalition_df(GLOVE), shapley_df({"L1": 1 / 6, "R": 2 / 3}))
    st = statuses(report)
    assert st[3] == FAIL and st[5] == FAIL
    assert st[9] == SKIPPED


def test_duplicate_channel_detected():
    sv = pd.DataFrame({"channel": ["L1", "L1", "L2", "R"], "shapley_value": [1 / 6, 1 / 6, 1 / 6, 2 / 3]})
    st = statuses(validate_shapley_outputs(coalition_df(GLOVE), sv))
    assert st[4] == FAIL


def test_unexpected_channel_detected():
    sv = shapley_df({**GLOVE_PHI, "TikTok": 0.0})
    st = statuses(validate_shapley_outputs(coalition_df(GLOVE), sv))
    assert st[6] == FAIL and st[3] == FAIL


def test_non_numeric_shapley_value_detected():
    sv = shapley_df(GLOVE_PHI).astype({"shapley_value": object})
    sv.loc[0, "shapley_value"] = "n/a"
    st = statuses(validate_shapley_outputs(coalition_df(GLOVE), sv))
    assert st[7] == FAIL and st[8] == PASS and st[9] == SKIPPED


def test_nan_and_infinity_text_from_csv_is_reported_as_non_finite_not_non_numeric(tmp_path):
    coalition_df(GLOVE).to_csv(tmp_path / "cv.csv", index=False)
    for text in ["nan", "NaN", "inf", "-inf"]:
        shapley_df(GLOVE_PHI).to_csv(tmp_path / "sv.csv", index=False)
        sv = (tmp_path / "sv.csv").read_text().replace("0.6666666666666666", text)
        (tmp_path / "sv.csv").write_text(sv)
        st = statuses(validate_shapley_files(tmp_path / "cv.csv", tmp_path / "sv.csv"))
        assert st[8] == FAIL and st[7] == PASS, text


def test_numeric_parsing_is_exact_for_csv_text(tmp_path):
    """Regression: values read from CSV text must equal Python's float() of that text."""
    from shapley_engine.shapley_validation import _numeric_column

    text = ["0.16666666666666666", "0.6666666666666666", "0.1", "1e-17"]
    arr, garbage = _numeric_column(pd.Series(text, dtype=str))
    assert arr.tolist() == [float(t) for t in text]
    assert not garbage.any()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinite_shapley_values_detected(bad):
    sv = shapley_df(GLOVE_PHI)
    sv.loc[1, "shapley_value"] = bad
    st = statuses(validate_shapley_outputs(coalition_df(GLOVE), sv))
    assert st[8] == FAIL and st[7] == PASS


# --------------------------------------------------------------------------- #
# mathematical checks (9, 10, 12)
# --------------------------------------------------------------------------- #
def test_efficiency_violation_detected():
    st = statuses(validate_shapley_outputs(coalition_df(GLOVE), shapley_df({**GLOVE_PHI, "R": 0.7})))
    assert st[9] == FAIL and st[10] == FAIL


def test_reference_catches_wrong_values_that_still_satisfy_efficiency():
    swapped = {"L1": 2 / 3, "L2": 1 / 6, "R": 1 / 6}  # same sum, wrong attribution
    st = statuses(validate_shapley_outputs(coalition_df(GLOVE), shapley_df(swapped)))
    assert st[9] == PASS
    assert st[10] == FAIL


def test_symmetry_violation_detected():
    unfair = {"L1": 1 / 6 + 0.05, "L2": 1 / 6 - 0.05, "R": 2 / 3}
    st = statuses(validate_shapley_outputs(coalition_df(GLOVE), shapley_df(unfair)))
    assert st[9] == PASS
    assert st[12] == FAIL


def test_null_channel_with_nonzero_phi_detected():
    wrong = {"A": 0.4, "D": 0.1}  # D contributes nothing but is credited
    st = statuses(validate_shapley_outputs(coalition_df(NULL_GAME), shapley_df(wrong)))
    assert st[9] == PASS and st[12] == FAIL and st[10] == FAIL


def test_correct_null_channel_passes():
    report = validate_shapley_outputs(coalition_df(NULL_GAME), shapley_df({"A": 0.5, "D": 0.0}))
    assert report.passed, report.format()


# --------------------------------------------------------------------------- #
# optional production-function hook (checks 11 and 13)
# --------------------------------------------------------------------------- #
def test_correct_production_function_passes_hand_examples():
    def correct(channels, values):
        return reference_shapley_by_subsets(channels, values)

    report = validate_shapley_outputs(coalition_df(GLOVE), shapley_df(GLOVE_PHI), shapley_fn=correct)
    assert status(report, 11) == PASS and status(report, 13) == PASS


def test_wrong_production_function_is_caught():
    def equal_split(channels, values):
        share = (values[frozenset(channels)] - values[frozenset()]) / len(channels)
        return {c: share for c in channels}

    report = validate_shapley_outputs(coalition_df(GLOVE), shapley_df(GLOVE_PHI), shapley_fn=equal_split)
    assert status(report, 11) == FAIL


def test_nondeterministic_production_function_is_caught():
    rng = random.Random()

    def noisy(channels, values):
        return {c: rng.random() for c in channels}

    report = validate_shapley_outputs(coalition_df(GLOVE), shapley_df(GLOVE_PHI), shapley_fn=noisy)
    assert status(report, 13) == FAIL


def test_crashing_production_function_is_reported_not_raised():
    def broken(channels, values):
        raise RuntimeError("boom")

    report = validate_shapley_outputs(coalition_df(GLOVE), shapley_df(GLOVE_PHI), shapley_fn=broken)
    assert status(report, 11) == FAIL


# --------------------------------------------------------------------------- #
# schema handling
# --------------------------------------------------------------------------- #
def test_column_names_are_auto_detected_case_insensitively():
    cv = coalition_df(PROJECT_GAME).rename(columns={"coalition": "Coalition", "value": "V"})
    sv = shapley_df(PROJECT_PHI).rename(columns={"channel": "Channel", "shapley_value": "Shapley_Value"})
    assert validate_shapley_outputs(cv, sv).passed


def test_explicit_column_names_are_honoured():
    cv = coalition_df(PROJECT_GAME).rename(columns={"coalition": "subset_text", "value": "conv_prob"})
    sv = shapley_df(PROJECT_PHI).rename(columns={"channel": "ch", "shapley_value": "attr"})
    report = validate_shapley_outputs(
        cv, sv, coalition_col="subset_text", value_col="conv_prob", channel_col="ch", shapley_col="attr"
    )
    assert report.passed


def test_unrecognised_columns_raise_schema_error():
    with pytest.raises(SchemaError):
        validate_shapley_outputs(pd.DataFrame({"x": [1], "y": [2]}), shapley_df(PROJECT_PHI))


def test_missing_input_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_shapley_files(tmp_path / "coalition_values.csv", tmp_path / "shapley_values.csv")


# --------------------------------------------------------------------------- #
# runner script
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("run_stage9", ROOT / "scripts" / "run_stage9.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, values, phi):
    coalition_df(values).to_csv(tmp_path / "cv.csv", index=False)
    shapley_df(phi).to_csv(tmp_path / "sv.csv", index=False)
    return ["--coalition-values", str(tmp_path / "cv.csv"), "--shapley-values", str(tmp_path / "sv.csv"),
            "--report-out", str(tmp_path / "report.csv")]


def test_runner_exit_zero_and_report_written_on_valid_files(runner, tmp_path, capsys):
    assert runner.main(_write(tmp_path, GLOVE, GLOVE_PHI)) == 0
    out = capsys.readouterr().out
    assert "VALIDATION PASSED" in out
    assert "INPUT STATUS: custom files" in out and "REAL-DATA STATUS: project files" not in out
    saved = pd.read_csv(tmp_path / "report.csv")
    assert len(saved) == 13 and set(saved["status"]) == {"PASS"}


def test_runner_exit_one_on_validation_failure(runner, tmp_path, capsys):
    assert runner.main(_write(tmp_path, GLOVE, {**GLOVE_PHI, "R": 0.7})) == 1
    assert "VALIDATION FAILED" in capsys.readouterr().out


def test_runner_reports_blocked_when_upstream_files_missing(runner, tmp_path, capsys):
    code = runner.main(["--coalition-values", str(tmp_path / "nope.csv"), "--shapley-values", str(tmp_path / "nope2.csv")])
    out = capsys.readouterr().out
    assert code == 2
    assert "BLOCKED" in out and "REAL-DATA STATUS: BLOCKED" in out
    assert not (tmp_path / "report.csv").exists()
