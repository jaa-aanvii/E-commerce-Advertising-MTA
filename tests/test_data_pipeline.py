"""Stage 2 tests.

IMPORTANT: the small CSV strings below are TEST FIXTURES, invented only so we can trigger specific
validation failures that the real Kaggle file does not contain. They are not project data and
never feed any result. The last test runs on the real file.
"""
from pathlib import Path

import pandas as pd
import pytest

from mta.data_pipeline import (
    KAGGLE_CONFIG,
    NO_CAMPAIGN,
    STANDARD_CONFIG,
    SchemaError,
    clean_events,
)

HEADER = "event_id,user_id,timestamp,channel,campaign,conversion,revenue\n"
GOOD_ROWS = [
    "e1,u1,2026-01-05T09:00:00Z,Email,promo,0,",
    "e2,u1,2026-01-06T10:00:00Z,Search Ad,kw,1,50.0",
]


def run(tmp_path, rows, header=HEADER, config=STANDARD_CONFIG):
    path = tmp_path / "events.csv"
    path.write_text(header + "\n".join(rows) + "\n")
    return clean_events(path, config)


def reasons_for(result, event_id):
    row = result.rejected[result.rejected["event_id"] == event_id]
    assert len(row) == 1, f"{event_id} was not rejected exactly once"
    return row["reject_reasons"].iloc[0]


# ---------------------------------------------------------------- valid data
def test_valid_data_is_fully_kept_with_correct_types(tmp_path):
    result = run(tmp_path, GOOD_ROWS)
    assert result.summary.rows_read == 2
    assert result.summary.rows_clean == 2
    assert result.summary.rows_rejected == 0
    assert len(result.rejected) == 0
    assert pd.api.types.is_datetime64_any_dtype(result.clean["timestamp"])
    assert list(result.clean["conversion"]) == [0, 1]
    assert result.clean["revenue"].iloc[1] == 50.0
    assert pd.isna(result.clean["revenue"].iloc[0])  # blank revenue stays missing, not invented


# ---------------------------------------------------------------- schema
def test_missing_required_column_raises(tmp_path):
    header = "event_id,user_id,timestamp,campaign,conversion,revenue\n"  # no channel column
    with pytest.raises(SchemaError):
        run(tmp_path, ["e1,u1,2026-01-05T09:00:00Z,promo,0,"], header=header)


# ---------------------------------------------------------------- invalid values: rejected and reported
def test_invalid_timestamps_are_rejected_with_reason(tmp_path):
    rows = GOOD_ROWS + [
        "e3,u2,not-a-date,Email,,0,",
        "e4,u2,2026-13-45T00:00:00Z,Email,,0,",  # impossible calendar date
        "e5,u2,,Email,,0,",  # blank
    ]
    result = run(tmp_path, rows)
    assert result.summary.rows_rejected == 3
    for eid in ("e3", "e4", "e5"):
        assert "invalid_timestamp" in reasons_for(result, eid)


def test_missing_user_id_is_rejected(tmp_path):
    result = run(tmp_path, GOOD_ROWS + ["e3,,2026-01-05T09:00:00Z,Email,,0,"])
    assert "missing_user_id" in reasons_for(result, "e3")


def test_missing_channel_is_rejected(tmp_path):
    result = run(tmp_path, GOOD_ROWS + ["e3,u2,2026-01-05T09:00:00Z,,,0,"])
    assert "missing_channel" in reasons_for(result, "e3")


@pytest.mark.parametrize("name", ["Conversion", "null", "START"])
def test_model_state_names_cannot_be_channels(tmp_path, name):
    result = run(tmp_path, GOOD_ROWS + [f"e3,u2,2026-01-05T09:00:00Z,{name},,0,"])
    assert "reserved_channel_name" in reasons_for(result, "e3")


@pytest.mark.parametrize("bad", ["maybe", "2", "-1", ""])
def test_invalid_conversion_flag_is_rejected(tmp_path, bad):
    result = run(tmp_path, GOOD_ROWS + [f"e3,u2,2026-01-05T09:00:00Z,Email,,{bad},"])
    assert "invalid_conversion_flag" in reasons_for(result, "e3")


def test_invalid_revenue_values_are_rejected(tmp_path):
    rows = GOOD_ROWS + [
        "e3,u2,2026-01-05T09:00:00Z,Email,,1,abc",  # not a number
        "e4,u2,2026-01-06T09:00:00Z,Email,,1,-5",  # negative
        "e5,u2,2026-01-07T09:00:00Z,Email,,0,20",  # money on a non-conversion
        "e6,u2,2026-01-08T09:00:00Z,Email,,1,inf",  # not finite
    ]
    result = run(tmp_path, rows)
    assert "invalid_revenue_not_numeric" in reasons_for(result, "e3")
    assert "invalid_revenue_negative" in reasons_for(result, "e4")
    assert "invalid_revenue_on_non_conversion" in reasons_for(result, "e5")
    assert "invalid_revenue_not_numeric" in reasons_for(result, "e6")


def test_zero_revenue_on_non_conversion_is_allowed(tmp_path):
    result = run(tmp_path, GOOD_ROWS + ["e3,u2,2026-01-05T09:00:00Z,Email,,0,0"])
    assert result.summary.rows_rejected == 0


def test_one_row_can_have_several_reasons(tmp_path):
    result = run(tmp_path, GOOD_ROWS + ["e3,,bad-date,Conversion,,maybe,"])
    reasons = reasons_for(result, "e3")
    for expected in ("missing_user_id", "invalid_timestamp", "reserved_channel_name", "invalid_conversion_flag"):
        assert expected in reasons


# ---------------------------------------------------------------- duplicate event IDs
def test_exact_duplicate_event_id_keeps_first_and_rejects_copy(tmp_path):
    rows = GOOD_ROWS + [GOOD_ROWS[0]]  # e1 logged twice, identical
    result = run(tmp_path, rows)
    assert result.summary.rows_clean == 2
    assert result.summary.rows_rejected == 1
    assert "duplicate_event_id_exact" in reasons_for(result, "e1")
    assert (result.clean["event_id"] == "e1").sum() == 1


def test_conflicting_duplicate_event_id_rejects_every_copy(tmp_path):
    rows = GOOD_ROWS + ["e1,u9,2026-02-01T09:00:00Z,Social,x,0,"]  # same id, different content
    result = run(tmp_path, rows)
    assert (result.rejected["event_id"] == "e1").sum() == 2
    assert "e1" not in set(result.clean["event_id"])
    assert result.summary.check_counts["duplicate_event_id_conflict"] == 2


# ---------------------------------------------------------------- legitimate repeats are NOT errors
def test_repeated_channel_touches_are_kept_and_not_flagged(tmp_path):
    rows = [
        "e1,u1,2026-01-05T09:00:00Z,Social,ig,0,",
        "e2,u1,2026-01-05T20:00:00Z,Social,ig,0,",  # same user, same channel, later time
        "e3,u1,2026-01-06T09:00:00Z,Social,ig,1,10",
    ]
    result = run(tmp_path, rows)
    assert result.summary.rows_clean == 3
    assert result.summary.rows_rejected == 0
    assert result.summary.repeated_channel_touches == 2
    assert not result.clean["tie_same_user_timestamp"].any()
    assert not result.clean["possible_duplicate_content"].any()


def test_same_user_same_timestamp_is_flagged_but_kept(tmp_path):
    rows = [
        "e1,u1,2026-01-05T09:00:00Z,Email,a,1,5",
        "e2,u1,2026-01-05T09:00:00Z,Search Ad,b,1,5",
    ]
    result = run(tmp_path, rows)
    assert result.summary.rows_rejected == 0
    assert result.summary.flagged_tie_rows == 2
    assert result.summary.flagged_possible_duplicate_rows == 0  # different channels


def test_identical_content_with_different_ids_is_flagged_not_deleted(tmp_path):
    rows = [
        "e1,u1,2026-01-05T09:00:00Z,Email,a,0,",
        "e2,u1,2026-01-05T09:00:00Z,Email,a,0,",
    ]
    result = run(tmp_path, rows)
    assert result.summary.rows_clean == 2
    assert result.summary.flagged_possible_duplicate_rows == 2


# ---------------------------------------------------------------- sorting
def test_sorting_is_by_user_then_timestamp_then_event_id(tmp_path):
    rows = [
        "e9,u2,2026-01-05T09:00:00Z,Email,,0,",
        "e5,u1,2026-01-07T09:00:00Z,Email,,0,",
        "e9x,u1,2026-01-05T09:00:00Z,Social,,0,",
        "e2x,u1,2026-01-05T09:00:00Z,Direct,,0,",  # ties with e9x: lower event_id goes first
    ]
    result = run(tmp_path, rows)
    assert list(result.clean["event_id"]) == ["e2x", "e9x", "e5", "e9"]


def test_sorting_does_not_depend_on_input_order(tmp_path):
    rows = GOOD_ROWS + ["e3,u2,2026-01-05T09:00:00Z,Email,,0,", "e4,u2,2026-01-05T09:00:00Z,Social,,0,"]
    forward = run(tmp_path, rows).clean["event_id"].tolist()
    backward = run(tmp_path, list(reversed(rows))).clean["event_id"].tolist()
    assert forward == backward


# ---------------------------------------------------------------- no silent loss
def test_every_row_is_accounted_for(tmp_path):
    rows = GOOD_ROWS + ["e3,,bad,Email,,0,", "e4,u3,2026-01-05T09:00:00Z,,,maybe,", GOOD_ROWS[1]]
    result = run(tmp_path, rows)
    s = result.summary
    assert s.rows_read == 5
    assert s.rows_read == s.rows_clean + s.rows_rejected
    assert len(result.rejected) == s.rows_rejected
    assert (result.rejected["reject_reasons"] != "").all()
    assert "file_row" in result.rejected.columns  # you can find each bad row in the original file


# ---------------------------------------------------------------- Kaggle-format mapping
def test_kaggle_format_is_mapped_and_event_id_derived(tmp_path):
    text = (
        "User ID,Timestamp,Channel,Campaign,Conversion\n"
        "5,10-02-2025 08:00,Email,-,No\n"
        "5,10-02-2025 09:30,Direct Traffic,Winter Sale,Yes\n"
    )
    path = tmp_path / "kaggle_style.csv"
    path.write_text(text)
    result = clean_events(path, KAGGLE_CONFIG)
    assert list(result.clean["event_id"]) == ["r00000001", "r00000002"]
    assert list(result.clean["campaign"]) == [NO_CAMPAIGN, "Winter Sale"]
    assert list(result.clean["conversion"]) == [0, 1]
    assert result.clean["timestamp"].iloc[0] == pd.Timestamp("2025-02-10 08:00")  # day-first parse
    assert "revenue" not in result.clean.columns  # absent in the source, so not invented
    assert result.summary.event_id_derived is True
    assert result.summary.revenue_present is False


# ---------------------------------------------------------------- the real file
REAL_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "multi_touch_attribution_data.csv"


@pytest.mark.skipif(not REAL_FILE.exists(), reason="real Kaggle file not present in data/raw/")
def test_real_file_accounts_for_every_row():
    result = clean_events(REAL_FILE, KAGGLE_CONFIG)
    s = result.summary
    assert s.rows_read == 10000
    assert s.rows_read == s.rows_clean + s.rows_rejected
    assert result.clean["event_id"].is_unique
    assert result.clean.groupby("user_id")["timestamp"].apply(lambda t: t.is_monotonic_increasing).all()
