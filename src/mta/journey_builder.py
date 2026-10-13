"""Stage 3: reconstruct customer journeys from validated events.

Scope: journeys only. NO transition counts, probabilities or removal effects (later stages).

Journey policy (the whole model rests on this, so it is stated once, here):

 1. Events are ordered per user by (timestamp, event_id).
 2. A journey is a run of one user's events that ends at that user's first conversion event
    (inclusive), or at the user's last event if no conversion follows.
 3. The conversion event is the journey's FINAL touch: it carries the channel the purchase
    happened through, and the journey then terminates in the state `Conversion`.
 4. The next event after a conversion starts a NEW journey for the same user (journey_seq + 1).
    A user therefore has as many journeys as conversions, plus one trailing Null journey if
    events follow the last conversion.
 5. Attribution window (optional, `window_hours`): a touch is eligible only if it happened within
    the window before the journey's final event. Older touches are excluded from the path, but
    counted in `n_touches_excluded_window`, never silently lost. A previous conversion is always
    a hard boundary: touches before it can never be credited to a later purchase.
 6. Segment: a journey is `returning` if the user had at least one conversion BEFORE the journey
    started, else `new`. Computed only from earlier events, so a journey's own conversion
    (or anything later) can never leak into its segment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mta.data_pipeline import DEFAULT_RAW_PATH, KAGGLE_CONFIG, clean_events, sort_events

START = "Start"
CONVERSION = "Conversion"
NULL = "Null"
PATH_SEP = " > "
DEFAULT_OUTPUT_PATH = Path("data/processed/journeys.csv")
REQUIRED_COLUMNS = ("event_id", "user_id", "timestamp", "channel", "conversion")
TIE_FLAG = "tie_same_user_timestamp"

JOURNEY_COLUMNS = [
    "journey_id", "user_id", "journey_seq", "segment", "prior_conversions", "converted",
    "terminal_state", "n_touches", "n_touches_excluded_window", "start_time", "end_time",
    "conversion_event_id", "has_timestamp_tie", "path",
]


@dataclass
class JourneySummary:
    window_hours: object
    n_events: int
    n_users: int
    n_journeys: int
    n_converting: int
    n_null: int
    n_new: int
    n_returning: int
    conversions_new: int
    conversions_returning: int
    conversion_events_in_data: int
    n_touches_in_paths: int
    n_touches_excluded: int
    single_touch_journeys: int
    mean_touches: float
    max_touches: int
    journeys_with_ties: int
    notes: list = field(default_factory=list)

    @property
    def event_accounting_ok(self) -> bool:
        return self.n_events == self.n_touches_in_paths + self.n_touches_excluded

    @property
    def conversion_accounting_ok(self) -> bool:
        return self.n_converting == self.conversion_events_in_data

    def to_text(self) -> str:
        lines = [
            "=== JOURNEY SUMMARY ===",
            f"Attribution window (hours): {self.window_hours if self.window_hours is not None else 'none'}",
            f"Events in:                 {self.n_events}",
            f"Users:                     {self.n_users}",
            f"Journeys:                  {self.n_journeys}",
            f"  ending in Conversion:    {self.n_converting}",
            f"  ending in Null:          {self.n_null}",
            f"  new customers:           {self.n_new}  (conversions: {self.conversions_new})",
            f"  returning customers:     {self.n_returning}  (conversions: {self.conversions_returning})",
            "",
            f"Touches in journey paths:  {self.n_touches_in_paths}",
            f"Touches excluded (window): {self.n_touches_excluded}",
            f"Event accounting (events == in paths + excluded): {self.event_accounting_ok}",
            f"Conversion accounting (converting journeys == conversion events): "
            f"{self.conversion_accounting_ok}",
            "",
            f"Touches per journey: mean {self.mean_touches:.2f}, max {self.max_touches}, "
            f"single-touch journeys {self.single_touch_journeys}",
            f"Journeys containing a same-minute tie: {self.journeys_with_ties}",
        ]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class JourneyResult:
    journeys: pd.DataFrame  # one row per journey
    touches: pd.DataFrame  # one row per event, with its journey_id and window eligibility
    summary: JourneySummary


# --------------------------------------------------------------------------- input check
def check_events(events: pd.DataFrame) -> None:
    """Fail loudly if the input is not the validated output of Stage 2."""
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events is missing required column(s): {missing}")
    if events["timestamp"].isna().any():
        raise ValueError("events contains missing timestamps; run clean_events first")
    if not events["conversion"].isin([0, 1]).all():
        raise ValueError("conversion must be 0 or 1 in every row; run clean_events first")
    if events["channel"].str.contains(PATH_SEP, regex=False).any():
        raise ValueError(f"a channel name contains the path separator {PATH_SEP!r}")
    reserved = {START.lower(), CONVERSION.lower(), NULL.lower()}
    if events["channel"].str.lower().isin(reserved).any():
        raise ValueError("a channel is named Start, Conversion or Null; these are model states")


# --------------------------------------------------------------------------- 1. journey ids
def assign_journey_ids(events: pd.DataFrame) -> pd.DataFrame:
    """Sort events and label each with journey_seq, journey_id and prior_conversions.

    Key idea: prior_conversions of an event = conversions by the same user at EARLIER positions in
    the sorted order (cumulative sum minus the event's own flag). A conversion event therefore
    belongs to the journey it ends, and the very next event starts a new one. Only past rows are
    used, so no future information can leak in.
    """
    df = sort_events(events)
    df["user_id"] = df["user_id"].astype(str)
    df["prior_conversions"] = (df.groupby("user_id")["conversion"].cumsum() - df["conversion"]).astype(int)
    df["journey_seq"] = df["prior_conversions"] + 1
    df["journey_id"] = df["user_id"] + "#" + df["journey_seq"].astype(str)
    return df


# --------------------------------------------------------------------------- 2. window
def apply_attribution_window(df: pd.DataFrame, window_hours) -> pd.DataFrame:
    """Mark each event `in_window`: within `window_hours` before its journey's final event.

    window_hours=None means no window (every event in the journey is eligible). The final event
    is always eligible (0 hours before itself), so no journey can become empty. The boundary is
    inclusive: a touch exactly `window_hours` before the end is eligible.
    """
    if window_hours is not None and window_hours <= 0:
        raise ValueError("window_hours must be positive or None")
    out = df.copy()
    end = out.groupby("journey_id")["timestamp"].transform("max")
    out["journey_end_time"] = end
    out["hours_before_end"] = (end - out["timestamp"]).dt.total_seconds() / 3600
    if window_hours is None:
        out["in_window"] = True
    else:
        out["in_window"] = (end - out["timestamp"]) <= pd.Timedelta(hours=window_hours)
    return out


# --------------------------------------------------------------------------- 3. paths
def build_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse eligible touches into one row per journey with an explicit Start/terminal path.

    The terminal state comes from the journey's last event: Conversion if it is a conversion
    event, otherwise Null. Start and the terminal state are added around the channel sequence;
    they are never read from, or written into, the channel column.
    """
    touches = df[df["in_window"]]
    grouped = touches.groupby("journey_id", sort=False)
    out = grouped.agg(
        user_id=("user_id", "first"),
        journey_seq=("journey_seq", "first"),
        prior_conversions=("prior_conversions", "first"),
        n_touches=("event_id", "size"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        last_conversion=("conversion", "last"),
        last_event_id=("event_id", "last"),
        channels=("channel", lambda s: PATH_SEP.join(s)),
    )
    if TIE_FLAG in touches.columns:
        out["has_timestamp_tie"] = grouped[TIE_FLAG].any()
    else:
        out["has_timestamp_tie"] = False
    out["n_touches_excluded_window"] = (~df["in_window"]).groupby(df["journey_id"]).sum().reindex(out.index)
    out["converted"] = out["last_conversion"].astype(int)
    out["terminal_state"] = np.where(out["converted"] == 1, CONVERSION, NULL)
    out["conversion_event_id"] = np.where(out["converted"] == 1, out["last_event_id"], "")
    out["path"] = START + PATH_SEP + out["channels"] + PATH_SEP + out["terminal_state"]
    return out.reset_index()


# --------------------------------------------------------------------------- 4. segment
def assign_segment(journeys: pd.DataFrame) -> pd.DataFrame:
    """new = no conversion before the journey started; returning = at least one.

    LIMITATION: we only see 2 days, so a purchase before the data starts is invisible and
    'new' really means 'no earlier conversion observed in this dataset'.
    """
    out = journeys.copy()
    out["segment"] = np.where(out["prior_conversions"] > 0, "returning", "new")
    return out


# --------------------------------------------------------------------------- 5. summary
def summarize_journeys(journeys: pd.DataFrame, touches: pd.DataFrame, window_hours) -> JourneySummary:
    converting = journeys["converted"] == 1
    returning = journeys["segment"] == "returning"
    notes = [
        "A conversion event is the journey's final touch; its channel is part of the path.",
        "Segments only reflect conversions seen in this dataset (see assign_segment).",
    ]
    n_tied_conv = int(((journeys["has_timestamp_tie"]) & converting).sum())
    if n_tied_conv:
        notes.append(
            f"{n_tied_conv} converting journey(s) contain a same-minute tie. Order (and so which "
            "channel ends the journey) follows event_id order, which is an arbitrary tiebreak."
        )
    return JourneySummary(
        window_hours=window_hours,
        n_events=len(touches),
        n_users=int(journeys["user_id"].nunique()),
        n_journeys=len(journeys),
        n_converting=int(converting.sum()),
        n_null=int((~converting).sum()),
        n_new=int((~returning).sum()),
        n_returning=int(returning.sum()),
        conversions_new=int((converting & ~returning).sum()),
        conversions_returning=int((converting & returning).sum()),
        conversion_events_in_data=int(touches["conversion"].sum()),
        n_touches_in_paths=int(journeys["n_touches"].sum()),
        n_touches_excluded=int(journeys["n_touches_excluded_window"].sum()),
        single_touch_journeys=int((journeys["n_touches"] == 1).sum()),
        mean_touches=float(journeys["n_touches"].mean()) if len(journeys) else 0.0,
        max_touches=int(journeys["n_touches"].max()) if len(journeys) else 0,
        journeys_with_ties=int(journeys["has_timestamp_tie"].sum()),
        notes=notes,
    )


# --------------------------------------------------------------------------- 6. orchestrate
def build_journeys(events: pd.DataFrame, window_hours=None) -> JourneyResult:
    """Validated events in, journeys out. `events` is the `clean` frame from Stage 2."""
    check_events(events)
    labelled = assign_journey_ids(events)
    windowed = apply_attribution_window(labelled, window_hours)
    journeys = assign_segment(build_paths(windowed))[JOURNEY_COLUMNS]
    touches_cols = ["journey_id", "journey_seq", "event_id", "user_id", "timestamp", "channel",
                    "conversion", "in_window", "hours_before_end"]
    if TIE_FLAG in windowed.columns:
        touches_cols.append(TIE_FLAG)
    touches = windowed[touches_cols]
    return JourneyResult(journeys, touches, summarize_journeys(journeys, touches, window_hours))


# --------------------------------------------------------------------------- 7. export
def export_journeys(journeys: pd.DataFrame, path=DEFAULT_OUTPUT_PATH) -> Path:
    """Write journeys to CSV, creating the folder if needed. Overwrites the previous export,
    which is fine: it is derived output and can be regenerated from the raw file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    journeys.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- manual verification
def show_user(result: JourneyResult, user_id: str) -> str:
    """Text report of one user: their sorted events, then the journeys built from them."""
    uid = str(user_id)
    t = result.touches[result.touches["user_id"] == uid]
    j = result.journeys[result.journeys["user_id"] == uid]
    if t.empty:
        return f"User {uid}: not found in the validated events."
    cols = ["event_id", "timestamp", "channel", "conversion", "journey_id", "in_window"]
    lines = [f"--- user {uid}: events (sorted) ---", t[cols].to_string(index=False),
             f"--- user {uid}: journeys ---"]
    for _, r in j.iterrows():
        lines.append(f"{r['journey_id']}  [{r['segment']}, prior_conversions={r['prior_conversions']}]  {r['path']}")
    return "\n".join(lines)


def main() -> None:
    validated = clean_events(DEFAULT_RAW_PATH, KAGGLE_CONFIG)
    if validated.summary.rows_rejected:
        print(f"WARNING: {validated.summary.rows_rejected} row(s) were rejected in Stage 2 and are not in any journey.")

    result = build_journeys(validated.clean, window_hours=None)
    out_path = export_journeys(result.journeys, DEFAULT_OUTPUT_PATH)
    print(result.summary.to_text())
    print(f"\nExported {len(result.journeys)} journeys to {out_path}")

    print("\n=== MANUAL VERIFICATION USERS ===")
    for uid in ("10062", "10167", "10068", "21704"):
        print(show_user(result, uid))
        print()

    print("=== WINDOW SENSITIVITY (not exported) ===")
    for hours in (None, 24, 6):
        s = build_journeys(validated.clean, window_hours=hours).summary
        label = "none" if hours is None else f"{hours}h"
        print(f"window {label:>4}: touches in paths {s.n_touches_in_paths}, excluded {s.n_touches_excluded}, "
              f"mean touches/journey {s.mean_touches:.2f}")


if __name__ == "__main__":
    main()
