"""Stage 3 tests.

The CSV rows below are TEST FIXTURES, invented to isolate one rule at a time. They are not project
data and never feed any result. The last tests run on the real Kaggle file.
"""
from pathlib import Path

import pandas as pd
import pytest

from mta.data_pipeline import KAGGLE_CONFIG, STANDARD_CONFIG, clean_events
from mta.journey_builder import (
    PATH_SEP,
    build_journeys,
    check_events,
    export_journeys,
)

HEADER = "event_id,user_id,timestamp,channel,campaign,conversion,revenue\n"


def build(tmp_path, rows, window_hours=None):
    path = tmp_path / "events.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n")
    validated = clean_events(path, STANDARD_CONFIG)
    assert validated.summary.rows_rejected == 0, "fixture rows must be valid"
    return build_journeys(validated.clean, window_hours=window_hours)


def ev(eid, user, ts, channel, conv=0):
    """One fixture row. ts is 'DD HH:MM' within January 2026."""
    day, hm = ts.split()
    return f"{eid},{user},2026-01-{day}T{hm}:00Z,{channel},,{conv},"


def path_of(result, journey_id):
    return result.journeys.set_index("journey_id").loc[journey_id, "path"]


# ---------------------------------------------------------------- ordering
def test_touches_are_sorted_chronologically_regardless_of_input_order(tmp_path):
    rows = [
        ev("e3", "u1", "07 09:00", "Direct", 1),
        ev("e1", "u1", "05 09:00", "Email"),
        ev("e2", "u1", "06 09:00", "Search Ad"),
    ]
    r = build(tmp_path, rows)
    assert path_of(r, "u1#1") == PATH_SEP.join(["Start", "Email", "Search Ad", "Direct", "Conversion"])


def test_same_timestamp_ties_follow_event_id_and_are_flagged(tmp_path):
    rows = [ev("e9", "u1", "05 09:00", "Social"), ev("e2", "u1", "05 09:00", "Email")]
    r = build(tmp_path, rows)
    assert path_of(r, "u1#1") == "Start > Email > Social > Null"  # e2 before e9
    assert bool(r.journeys.loc[0, "has_timestamp_tie"]) is True


def test_journey_order_is_stable_across_input_order(tmp_path):
    rows = [ev("e1", "u1", "05 09:00", "Email"), ev("e2", "u1", "05 09:00", "Social"),
            ev("e3", "u2", "05 10:00", "Direct", 1)]
    a = build(tmp_path, rows).journeys["path"].tolist()
    b = build(tmp_path, list(reversed(rows))).journeys["path"].tolist()
    assert a == b


# ---------------------------------------------------------------- repeated channels
def test_repeated_channels_are_preserved(tmp_path):
    rows = [ev("e1", "u1", "05 09:00", "Social"), ev("e2", "u1", "05 20:00", "Social"),
            ev("e3", "u1", "06 09:00", "Search Ad")]
    r = build(tmp_path, rows)
    assert path_of(r, "u1#1") == "Start > Social > Social > Search Ad > Null"
    assert r.journeys.loc[0, "n_touches"] == 3


# ---------------------------------------------------------------- Start and terminal states
def test_start_and_terminal_states(tmp_path):
    rows = [
        ev("e1", "u1", "05 09:00", "Email"), ev("e2", "u1", "06 09:00", "Direct", 1),  # converts
        ev("e3", "u2", "05 09:00", "Email"),  # never converts
    ]
    j = build(tmp_path, rows).journeys.set_index("journey_id")
    assert j.loc["u1#1", "path"] == "Start > Email > Direct > Conversion"
    assert j.loc["u1#1", "terminal_state"] == "Conversion" and j.loc["u1#1", "converted"] == 1
    assert j.loc["u2#1", "path"] == "Start > Email > Null"
    assert j.loc["u2#1", "terminal_state"] == "Null" and j.loc["u2#1", "converted"] == 0


def test_state_names_never_appear_as_touches(tmp_path):
    rows = [ev("e1", "u1", "05 09:00", "Email"), ev("e2", "u1", "06 09:00", "Direct", 1)]
    r = build(tmp_path, rows)
    assert not r.touches["channel"].str.lower().isin({"start", "conversion", "null"}).any()


def test_conversion_event_is_final_touch_and_its_channel_is_kept(tmp_path):
    rows = [ev("e1", "u1", "05 09:00", "Email"), ev("e2", "u1", "06 09:00", "Direct", 1)]
    j = build(tmp_path, rows).journeys.iloc[0]
    assert j["path"].endswith("Direct > Conversion")
    assert j["conversion_event_id"] == "e2"


def test_events_after_a_conversion_are_not_attributed_to_it(tmp_path):
    rows = [ev("e1", "u1", "05 09:00", "Email", 1), ev("e2", "u1", "06 09:00", "Social")]
    j = build(tmp_path, rows).journeys.set_index("journey_id")
    assert j.loc["u1#1", "path"] == "Start > Email > Conversion"  # Social is not in it
    assert j.loc["u1#2", "path"] == "Start > Social > Null"


# ---------------------------------------------------------------- repeated purchases
def test_one_user_with_several_conversions_gets_several_journeys(tmp_path):
    rows = [
        ev("e1", "u1", "05 09:00", "Email"), ev("e2", "u1", "05 10:00", "Direct", 1),
        ev("e3", "u1", "06 09:00", "Social"), ev("e4", "u1", "06 10:00", "Search Ad", 1),
        ev("e5", "u1", "07 09:00", "Email"),
    ]
    j = build(tmp_path, rows).journeys
    assert j["journey_id"].tolist() == ["u1#1", "u1#2", "u1#3"]
    assert j["path"].tolist() == [
        "Start > Email > Direct > Conversion",
        "Start > Social > Search Ad > Conversion",
        "Start > Email > Null",
    ]


def test_back_to_back_conversions_become_separate_single_touch_journeys(tmp_path):
    rows = [ev("e1", "u1", "05 09:00", "Email", 1), ev("e2", "u1", "05 10:00", "Direct", 1)]
    j = build(tmp_path, rows).journeys
    assert j["path"].tolist() == ["Start > Email > Conversion", "Start > Direct > Conversion"]


def test_each_conversion_event_is_counted_exactly_once(tmp_path):
    rows = [
        ev("e1", "u1", "05 09:00", "Email", 1), ev("e2", "u1", "05 10:00", "Direct", 1),
        ev("e3", "u2", "05 09:00", "Social"), ev("e4", "u2", "06 09:00", "Social", 1),
    ]
    r = build(tmp_path, rows)
    ids = r.journeys.loc[r.journeys["converted"] == 1, "conversion_event_id"]
    assert sorted(ids) == ["e1", "e2", "e4"]  # every conversion event, once
    assert ids.is_unique
    assert r.summary.conversion_accounting_ok


# ---------------------------------------------------------------- segmentation
def test_segments_new_then_returning(tmp_path):
    rows = [
        ev("e1", "u1", "05 09:00", "Email", 1),
        ev("e2", "u1", "06 09:00", "Social", 1),
        ev("e3", "u1", "07 09:00", "Direct"),
    ]
    j = build(tmp_path, rows).journeys
    assert j["segment"].tolist() == ["new", "returning", "returning"]
    assert j["prior_conversions"].tolist() == [0, 1, 2]


def test_user_who_never_converts_is_new(tmp_path):
    j = build(tmp_path, [ev("e1", "u1", "05 09:00", "Email")]).journeys
    assert j.loc[0, "segment"] == "new" and j.loc[0, "prior_conversions"] == 0


def test_own_conversion_does_not_leak_into_its_segment(tmp_path):
    # The first journey converts, but it must still be 'new': only EARLIER conversions count.
    j = build(tmp_path, [ev("e1", "u1", "05 09:00", "Email", 1)]).journeys
    assert j.loc[0, "segment"] == "new"


def test_segment_does_not_change_when_later_events_are_added(tmp_path):
    base = [ev("e1", "u1", "05 09:00", "Email", 1), ev("e2", "u1", "06 09:00", "Social")]
    later = base + [ev("e3", "u1", "07 09:00", "Direct", 1), ev("e4", "u1", "08 09:00", "Email", 1)]
    short = build(tmp_path, base).journeys.set_index("journey_id")
    long = build(tmp_path, later).journeys.set_index("journey_id")
    for jid in short.index:
        assert short.loc[jid, "prior_conversions"] == long.loc[jid, "prior_conversions"]
        assert short.loc[jid, "segment"] == long.loc[jid, "segment"]


def test_segments_are_tracked_per_user(tmp_path):
    rows = [ev("e1", "u1", "05 09:00", "Email", 1), ev("e2", "u2", "06 09:00", "Social")]
    j = build(tmp_path, rows).journeys.set_index("user_id")
    assert j.loc["u2", "segment"] == "new"  # u1's purchase says nothing about u2


# ---------------------------------------------------------------- attribution window
WINDOW_ROWS = [
    ev("e1", "u1", "05 00:00", "Display"),   # 30h before the end
    ev("e2", "u1", "05 12:00", "Social"),    # exactly 18h before
    ev("e3", "u1", "06 06:00", "Email", 1),  # the end
]


def test_no_window_keeps_every_touch(tmp_path):
    r = build(tmp_path, WINDOW_ROWS, window_hours=None)
    assert path_of(r, "u1#1") == "Start > Display > Social > Email > Conversion"
    assert r.summary.n_touches_excluded == 0


def test_window_excludes_old_touches_and_reports_them(tmp_path):
    r = build(tmp_path, WINDOW_ROWS, window_hours=24)
    assert path_of(r, "u1#1") == "Start > Social > Email > Conversion"
    assert r.journeys.loc[0, "n_touches_excluded_window"] == 1
    assert r.summary.event_accounting_ok  # nothing vanished: 3 events = 2 in path + 1 excluded
    assert r.touches.set_index("event_id").loc["e1", "in_window"] == False  # noqa: E712


def test_window_boundary_is_inclusive(tmp_path):
    r = build(tmp_path, WINDOW_ROWS, window_hours=18)  # e2 is exactly 18h before the end
    assert path_of(r, "u1#1") == "Start > Social > Email > Conversion"
    r2 = build(tmp_path, WINDOW_ROWS, window_hours=17.5)
    assert path_of(r2, "u1#1") == "Start > Email > Conversion"


def test_final_touch_is_always_kept_even_with_tiny_window(tmp_path):
    r = build(tmp_path, WINDOW_ROWS, window_hours=0.01)
    assert path_of(r, "u1#1") == "Start > Email > Conversion"


def test_previous_conversion_is_a_hard_boundary_for_attribution(tmp_path):
    rows = [
        ev("e1", "u1", "05 09:00", "Display", 1),   # first purchase
        ev("e2", "u1", "05 10:00", "Social"),
        ev("e3", "u1", "05 11:00", "Email", 1),     # second purchase
    ]
    j = build(tmp_path, rows, window_hours=None).journeys  # even with no window
    assert j["path"].tolist() == ["Start > Display > Conversion", "Start > Social > Email > Conversion"]


def test_window_applies_to_null_journeys_from_their_last_event(tmp_path):
    rows = [ev("e1", "u1", "05 00:00", "Email"), ev("e2", "u1", "07 00:00", "Social")]
    r = build(tmp_path, rows, window_hours=24)
    assert path_of(r, "u1#1") == "Start > Social > Null"


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_window_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        build(tmp_path, [ev("e1", "u1", "05 09:00", "Email")], window_hours=bad)


# ---------------------------------------------------------------- input checks
def test_missing_columns_raise():
    with pytest.raises(ValueError):
        check_events(pd.DataFrame({"user_id": ["u1"]}))


def test_channel_containing_path_separator_raises(tmp_path):
    df = pd.DataFrame({
        "event_id": ["e1"], "user_id": ["u1"], "timestamp": pd.to_datetime(["2026-01-05"]),
        "channel": ["Email > Promo"], "conversion": [0],
    })
    with pytest.raises(ValueError):
        build_journeys(df)


# ---------------------------------------------------------------- export
def test_export_writes_readable_csv_and_creates_folder(tmp_path):
    r = build(tmp_path, [ev("e1", "u1", "05 09:00", "Email"), ev("e2", "u1", "06 09:00", "Direct", 1)])
    out = export_journeys(r.journeys, tmp_path / "nested" / "journeys.csv")
    back = pd.read_csv(out)
    assert len(back) == len(r.journeys)
    assert back.loc[0, "path"] == "Start > Email > Direct > Conversion"
    assert {"journey_id", "segment", "terminal_state", "path"} <= set(back.columns)


# ---------------------------------------------------------------- the real file
REAL_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "multi_touch_attribution_data.csv"
needs_real = pytest.mark.skipif(not REAL_FILE.exists(), reason="real Kaggle file not present in data/raw/")


@needs_real
def test_real_file_accounting_and_known_journeys():
    r = build_journeys(clean_events(REAL_FILE, KAGGLE_CONFIG).clean)
    s = r.summary
    assert s.event_accounting_ok and s.conversion_accounting_ok
    assert s.n_events == 10000 and s.n_converting == 4944
    assert s.n_journeys == s.n_converting + s.n_null
    j = r.journeys.set_index("journey_id")
    assert j.loc["10062#1", "path"] == "Start > Social Media > Direct Traffic > Email > Conversion"
    assert j.loc["10167#1", "path"] == "Start > Social Media > Display Ads > Display Ads > Conversion"
    assert j.loc["10068#2", "path"] == "Start > Social Media > Social Media > Conversion"
    assert j.loc["10068#2", "segment"] == "returning"


@needs_real
def test_real_file_each_conversion_event_appears_in_exactly_one_journey():
    validated = clean_events(REAL_FILE, KAGGLE_CONFIG)
    r = build_journeys(validated.clean)
    ids = r.journeys.loc[r.journeys["converted"] == 1, "conversion_event_id"]
    assert ids.is_unique
    expected = set(validated.clean.loc[validated.clean["conversion"] == 1, "event_id"])
    assert set(ids) == expected


@needs_real
def test_real_file_window_never_loses_events():
    validated = clean_events(REAL_FILE, KAGGLE_CONFIG)
    for hours in (None, 24, 6):
        s = build_journeys(validated.clean, window_hours=hours).summary
        assert s.event_accounting_ok
