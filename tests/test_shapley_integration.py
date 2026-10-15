"""Tests for Task 1.5 - final Shapley integration.

ALL games and tables in this file are SYNTHETIC (hand-made fixtures for unit testing).
None of the numbers here are project results.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shapley_engine.shapley_integration import (  # noqa: E402
    FINAL_COLUMNS,
    NORM_OK,
    NORM_SIGNED,
    NORM_UNDEFINED,
    IntegrationBlockedError,
    build_final_table,
    check_task2_ready,
    integrate_shapley_files,
    integrate_shapley_frames,
    normalize_shapley,
    rank_channels,
)

F = frozenset

# --- synthetic games (phi derived by hand) ---
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
GLOVE_PHI = {"L1": 1 / 6, "L2": 1 / 6, "R": 2 / 3}  # sums to 1 = V(N) - V(empty)

NEGATIVE_GAME = {F(): 0.0, F("A"): 0.6, F("B"): 0.1, F("AB"): 0.4}
NEGATIVE_PHI = {"A": 0.45, "B": -0.05}  # total 0.4

# every channel lowers conversion: V(empty)=0.5 > V(N)=0.2 -> total phi = -0.3
HARMFUL_GAME = {F(): 0.5, F("A"): 0.3, F("B"): 0.4, F("AB"): 0.2}
HARMFUL_PHI = {"A": -0.2, "B": -0.1}

# V(empty) > 0: phi_A=0.4, phi_B=0.3, total 0.7 = 0.9 - 0.2
POSITIVE_EMPTY_GAME = {F(): 0.2, F("A"): 0.5, F("B"): 0.4, F("AB"): 0.9}
POSITIVE_EMPTY_PHI = {"A": 0.4, "B": 0.3}


def coalition_df(values):
    rows = [
        {"coalition": "|".join(sorted(k)), "value": v}
        for k, v in sorted(values.items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))
    ]
    return pd.DataFrame(rows)


def shapley_df(phi):
    return pd.DataFrame({"channel": list(phi), "shapley_value": list(phi.values())})


def write_inputs(tmp_path, values, phi, tag=""):
    cv, sv = tmp_path / f"cv{tag}.csv", tmp_path / f"sv{tag}.csv"
    coalition_df(values).to_csv(cv, index=False)
    shapley_df(phi).to_csv(sv, index=False)
    return cv, sv


# --------------------------------------------------------------------------- #
# 1-3. correct integration, channels complete, no duplicates
# --------------------------------------------------------------------------- #
def test_correct_final_integration(tmp_path):
    cv, sv = write_inputs(tmp_path, GLOVE, GLOVE_PHI)
    out, meta = tmp_path / "final.csv", tmp_path / "meta.json"
    result = integrate_shapley_files(cv, sv, out, meta)

    saved = pd.read_csv(out)
    assert list(saved.columns) == FINAL_COLUMNS
    assert result.meta["task2_ready"] is True
    assert result.validation_report.passed
    by_channel = saved.set_index("channel")
    assert by_channel.loc["R", "shapley_value"] == pytest.approx(2 / 3)
    assert by_channel.loc["L1", "shapley_value"] == pytest.approx(1 / 6)
    assert json.loads(meta.read_text())["validation"]["passed"] is True


def test_all_expected_channels_present_and_unique(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, GLOVE, GLOVE_PHI), tmp_path / "f.csv")
    channels = list(result.table["channel"])
    assert sorted(channels) == ["L1", "L2", "R"]
    assert len(channels) == len(set(channels))


def test_missing_channel_blocks_integration(tmp_path):
    cv, sv = write_inputs(tmp_path, GLOVE, {"L1": 1 / 6, "R": 2 / 3})
    out = tmp_path / "f.csv"
    with pytest.raises(IntegrationBlockedError):
        integrate_shapley_files(cv, sv, out)
    assert not out.exists()


def test_duplicate_channel_blocks_integration():
    sv = pd.DataFrame({"channel": ["L1", "L1", "L2", "R"], "shapley_value": [1 / 6, 1 / 6, 1 / 6, 2 / 3]})
    with pytest.raises(IntegrationBlockedError) as exc:
        integrate_shapley_frames(coalition_df(GLOVE), sv)
    assert any(r.number == 4 and r.status == "FAIL" for r in exc.value.report.results)


# --------------------------------------------------------------------------- #
# 4. Start / Conversion / Null are not channels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reserved", ["Start", "Conversion", "Null"])
def test_reserved_state_in_shapley_output_blocks_integration(reserved):
    with pytest.raises(IntegrationBlockedError) as exc:
        integrate_shapley_frames(coalition_df(GLOVE), shapley_df({**GLOVE_PHI, reserved: 0.0}))
    assert any(r.number == 6 and r.status == "FAIL" for r in exc.value.report.results)


def test_reserved_state_inside_coalition_blocks_integration():
    cv = coalition_df(GLOVE)
    cv.loc[cv["coalition"] == "R", "coalition"] = "R|Null"
    with pytest.raises(IntegrationBlockedError):
        integrate_shapley_frames(cv, shapley_df(GLOVE_PHI))


def test_final_table_has_no_reserved_states(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, GLOVE, GLOVE_PHI), tmp_path / "f.csv")
    assert not {"start", "conversion", "null"} & {c.lower() for c in result.table["channel"]}


# --------------------------------------------------------------------------- #
# 5. efficiency
# --------------------------------------------------------------------------- #
def test_efficiency_recorded_in_meta(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, POSITIVE_EMPTY_GAME, POSITIVE_EMPTY_PHI), tmp_path / "f.csv")
    meta = result.meta
    assert meta["total_shapley"] == pytest.approx(0.7)
    assert meta["expected_total"] == pytest.approx(0.9 - 0.2)
    assert meta["v_empty"] == pytest.approx(0.2)
    assert meta["efficiency_passed"] is True
    assert math.isclose(result.table["shapley_value"].sum(), meta["expected_total"], abs_tol=1e-9)


def test_efficiency_failure_blocks_and_does_not_modify_values(tmp_path):
    cv, sv = write_inputs(tmp_path, GLOVE, {**GLOVE_PHI, "R": 0.7})
    out = tmp_path / "f.csv"
    with pytest.raises(IntegrationBlockedError) as exc:
        integrate_shapley_files(cv, sv, out)
    assert any(r.number == 9 and r.status == "FAIL" for r in exc.value.report.results)
    assert not out.exists()


def test_task_1_4_validation_cannot_be_bypassed():
    """Efficient but wrong values (swapped attribution) are still caught by the Task 1.4 reference check."""
    swapped = {"L1": 2 / 3, "L2": 1 / 6, "R": 1 / 6}
    with pytest.raises(IntegrationBlockedError) as exc:
        integrate_shapley_frames(coalition_df(GLOVE), shapley_df(swapped))
    failed = {r.number for r in exc.value.report.results if r.status == "FAIL"}
    assert 10 in failed and 9 not in failed


# --------------------------------------------------------------------------- #
# 6. determinism
# --------------------------------------------------------------------------- #
def test_output_is_deterministic_and_input_order_independent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cv, sv = write_inputs(tmp_path, GLOVE, GLOVE_PHI)
    integrate_shapley_files(cv, sv, a / "f.csv", a / "m.json")
    integrate_shapley_files(cv, sv, b / "f.csv", b / "m.json")
    assert (a / "f.csv").read_bytes() == (b / "f.csv").read_bytes()
    assert (a / "m.json").read_bytes() == (b / "m.json").read_bytes()

    shuffled_cv = coalition_df(GLOVE).sample(frac=1, random_state=5).reset_index(drop=True)
    shuffled_sv = shapley_df(dict(reversed(list(GLOVE_PHI.items()))))
    c = tmp_path / "c"
    integrate_shapley_frames(shuffled_cv, shuffled_sv, output_path=c / "f.csv")
    assert (c / "f.csv").read_bytes() == (a / "f.csv").read_bytes()


# --------------------------------------------------------------------------- #
# 7. ranking
# --------------------------------------------------------------------------- #
def test_raw_shapley_values_are_bit_for_bit_unchanged_by_integration(tmp_path):
    """Regression: a 1-ulp parsing error once altered 0.16666666666666666 on its way through."""
    cv, sv = write_inputs(tmp_path, GLOVE, GLOVE_PHI)
    in_values = pd.read_csv(sv, dtype=str).set_index("channel")["shapley_value"]
    expected = {channel: float(text) for channel, text in in_values.items()}
    out = tmp_path / "f.csv"
    result = integrate_shapley_files(cv, sv, out)
    assert dict(zip(result.table["channel"], result.table["shapley_value"])) == expected  # exact equality
    saved = pd.read_csv(out, dtype=str).set_index("channel")["shapley_value"]
    assert {c: float(s) for c, s in saved.items()} == expected


def test_ranking_is_descending_with_alphabetical_tie_break():
    assert rank_channels({"B": 0.2, "A": 0.2, "C": 0.5, "D": -0.1}) == {"C": 1, "A": 2, "B": 3, "D": 4}


def test_ranking_treats_float_noise_as_a_tie():
    assert rank_channels({"Z": 1 / 6, "A": 0.16666666666666666 + 1e-15}) == {"A": 1, "Z": 2}


def test_final_table_is_sorted_by_rank(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, GLOVE, GLOVE_PHI), tmp_path / "f.csv")
    assert list(result.table["rank"]) == [1, 2, 3]
    assert list(result.table["channel"]) == ["R", "L1", "L2"]  # L1/L2 tie -> alphabetical


# --------------------------------------------------------------------------- #
# 8-9. normalized weights and percentages
# --------------------------------------------------------------------------- #
def test_normalized_weights_are_separate_from_raw_values(tmp_path):
    cv, sv = write_inputs(tmp_path, POSITIVE_EMPTY_GAME, POSITIVE_EMPTY_PHI)
    result = integrate_shapley_files(cv, sv, tmp_path / "f.csv")
    t = result.table.set_index("channel")
    # raw values are exactly what Task 1.3 produced (total 0.7, NOT rescaled to 1)
    assert t.loc["A", "shapley_value"] == 0.4 and t.loc["B", "shapley_value"] == 0.3
    # weights are phi / sum(phi)
    assert t.loc["A", "normalized_weight"] == pytest.approx(0.4 / 0.7)
    assert t.loc["B", "normalized_weight"] == pytest.approx(0.3 / 0.7)
    assert t["normalized_weight"].sum() == pytest.approx(1.0)
    assert result.meta["normalization_status"] == NORM_OK


def test_percentages_sum_to_100(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, GLOVE, GLOVE_PHI), tmp_path / "f.csv")
    t = result.table
    assert t["attribution_percentage"].sum() == pytest.approx(100.0)
    assert t["attribution_percentage"].tolist() == pytest.approx((t["normalized_weight"] * 100).tolist())
    assert t.set_index("channel").loc["R", "attribution_percentage"] == pytest.approx(200 / 3)


def test_normalize_shapley_policies():
    weights, status = normalize_shapley({"A": 0.4, "B": 0.6})
    assert status == NORM_OK and weights == {"A": pytest.approx(0.4), "B": pytest.approx(0.6)}
    weights, status = normalize_shapley({"A": 0.45, "B": -0.05})
    assert status == NORM_SIGNED and weights["B"] == pytest.approx(-0.125)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert normalize_shapley({"A": -0.2, "B": -0.1}) == (None, NORM_UNDEFINED)
    assert normalize_shapley({"A": 0.0, "B": 0.0}) == (None, NORM_UNDEFINED)


# --------------------------------------------------------------------------- #
# 10. negative Shapley values are not rejected, not altered
# --------------------------------------------------------------------------- #
def test_negative_shapley_value_is_kept_unchanged(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, NEGATIVE_GAME, NEGATIVE_PHI), tmp_path / "f.csv")
    t = result.table.set_index("channel")
    assert t.loc["B", "shapley_value"] == -0.05  # not abs(), not clipped to 0
    assert t.loc["A", "shapley_value"] == 0.45
    assert result.meta["has_negative_shapley_values"] is True
    assert result.meta["normalization_status"] == NORM_SIGNED
    assert t.loc["B", "normalized_weight"] == pytest.approx(-0.125)  # signed ratio, not clipped
    assert t["normalized_weight"].sum() == pytest.approx(1.0)
    assert result.meta["task2_ready"] is True
    assert any("negative" in w for w in result.meta["task2_warnings"])
    assert t.loc["B", "rank"] == 2


def test_non_positive_total_leaves_weights_undefined_but_keeps_raw_values(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, HARMFUL_GAME, HARMFUL_PHI), tmp_path / "f.csv")
    t = result.table.set_index("channel")
    assert result.meta["normalization_status"] == NORM_UNDEFINED
    assert t.loc["A", "shapley_value"] == -0.2 and t.loc["B", "shapley_value"] == -0.1
    assert t["normalized_weight"].isna().all()
    assert result.meta["task2_ready"] is False
    assert result.meta["total_shapley"] == pytest.approx(-0.3)


# --------------------------------------------------------------------------- #
# 11-14. missing / invalid input
# --------------------------------------------------------------------------- #
def test_missing_input_files_raise_clear_error(tmp_path):
    out = tmp_path / "f.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        integrate_shapley_files(tmp_path / "no_cv.csv", tmp_path / "no_sv.csv", out)
    assert not out.exists()
    cv, _ = write_inputs(tmp_path, GLOVE, GLOVE_PHI)
    with pytest.raises(FileNotFoundError, match="Task 1.3"):
        integrate_shapley_files(cv, tmp_path / "no_sv.csv", out)


@pytest.mark.parametrize("bad, failed_check", [("abc", 7), ("nan", 8), ("inf", 8), ("-inf", 8)])
def test_invalid_shapley_values_block_integration(tmp_path, bad, failed_check):
    cv, sv = write_inputs(tmp_path, GLOVE, GLOVE_PHI)
    text = sv.read_text().replace("0.6666666666666666", bad)
    sv.write_text(text)
    out = tmp_path / "f.csv"
    with pytest.raises(IntegrationBlockedError) as exc:
        integrate_shapley_files(cv, sv, out)
    assert any(r.number == failed_check and r.status == "FAIL" for r in exc.value.report.results)
    assert not out.exists()


def test_blocked_run_does_not_touch_an_existing_output_file(tmp_path):
    cv, sv = write_inputs(tmp_path, GLOVE, {**GLOVE_PHI, "R": 0.7})
    out = tmp_path / "f.csv"
    out.write_text("earlier run\n")
    with pytest.raises(IntegrationBlockedError):
        integrate_shapley_files(cv, sv, out)
    assert out.read_text() == "earlier run\n"


def test_invalid_coalition_value_blocks_integration():
    cv = coalition_df(GLOVE)
    cv["value"] = cv["value"].astype(object)
    cv.loc[cv["coalition"] == "L1|R", "value"] = float("nan")
    with pytest.raises(IntegrationBlockedError):
        integrate_shapley_frames(cv, shapley_df(GLOVE_PHI))


# --------------------------------------------------------------------------- #
# 15. Task 2 input compatibility
# --------------------------------------------------------------------------- #
def test_final_table_meets_task2_contract(tmp_path):
    result = integrate_shapley_files(*write_inputs(tmp_path, GLOVE, GLOVE_PHI), tmp_path / "f.csv")
    assert check_task2_ready(result.table) == ([], [])
    reloaded = pd.read_csv(tmp_path / "f.csv")  # what Task 2 will actually read
    assert list(reloaded.columns) == FINAL_COLUMNS
    assert check_task2_ready(reloaded) == ([], [])


def test_task2_contract_detects_problems():
    good, _ = build_final_table(GLOVE_PHI)
    assert check_task2_ready(good.drop(columns=["normalized_weight"]))[0]
    dup = good.copy()
    dup.loc[1, "channel"] = dup.loc[0, "channel"]
    assert any("duplicate" in p for p in check_task2_ready(dup)[0])
    reserved = good.copy()
    reserved.loc[0, "channel"] = "Null"
    assert any("reserved" in p for p in check_task2_ready(reserved)[0])
    skewed = good.copy()
    skewed.loc[0, "normalized_weight"] += 0.1
    assert any("sums to" in p for p in check_task2_ready(skewed)[0])
    nan_phi = good.copy()
    nan_phi.loc[0, "shapley_value"] = float("nan")
    assert check_task2_ready(nan_phi)[0]
    bad_rank = good.copy()
    bad_rank["rank"] = [1, 1, 2]
    assert any("rank" in p for p in check_task2_ready(bad_rank)[0])


# --------------------------------------------------------------------------- #
# runner script
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("run_stage10", ROOT / "scripts" / "run_stage10.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path, cv, sv):
    return [
        "--coalition-values", str(cv), "--shapley-values", str(sv),
        "--output", str(tmp_path / "final.csv"), "--meta-out", str(tmp_path / "meta.json"),
    ]


def test_runner_success_writes_outputs_and_reports(runner, tmp_path, capsys):
    cv, sv = write_inputs(tmp_path, GLOVE, GLOVE_PHI)
    assert runner.main(_argv(tmp_path, cv, sv)) == 0
    out = capsys.readouterr().out
    assert "Validation: PASSED (13/13 checks)" in out and "Efficiency: PASSED" in out
    assert "Integration: SUCCESS" in out and "INPUT STATUS: custom files" in out
    assert len(pd.read_csv(tmp_path / "final.csv")) == 3
    assert json.loads((tmp_path / "meta.json").read_text())["task2_ready"] is True


def test_runner_exit_1_when_validation_fails(runner, tmp_path, capsys):
    cv, sv = write_inputs(tmp_path, GLOVE, {**GLOVE_PHI, "R": 0.7})
    assert runner.main(_argv(tmp_path, cv, sv)) == 1
    out = capsys.readouterr().out
    assert "Integration: BLOCKED by validation failure" in out
    assert "Integration: SUCCESS" not in out
    assert not (tmp_path / "final.csv").exists()


def test_runner_exit_2_when_upstream_missing(runner, tmp_path, capsys):
    assert runner.main(_argv(tmp_path, tmp_path / "x.csv", tmp_path / "y.csv")) == 2
    out = capsys.readouterr().out
    assert "BLOCKED - required upstream output is missing" in out and "REAL-DATA STATUS: BLOCKED" in out
    assert not (tmp_path / "final.csv").exists()
