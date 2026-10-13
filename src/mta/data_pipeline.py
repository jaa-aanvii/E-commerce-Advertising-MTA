"""Stage 2: load and validate raw event data.

Scope: loading, validation, deterministic sorting. NO journey or Markov logic.

Policy: no row is ever silently dropped.
  * Invalid rows are moved to `rejected`, each with a human-readable reason.
  * Usable-but-questionable rows stay in `clean` with a flag column.
  * Every count is reported in the validation summary, including checks that found nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RAW_PATH = Path("data/raw/multi_touch_attribution_data.csv")
NO_CAMPAIGN = "(none)"
# Model states added in later stages. They must never appear as ordinary channels.
RESERVED_STATE_NAMES = {"start", "conversion", "null"}
# Fields every row must have. event_id is also required, but can be derived (see map_to_schema).
REQUIRED_FIELDS = ("user_id", "timestamp", "channel", "conversion")


class SchemaError(ValueError):
    """The file cannot be interpreted at all (for example, a required column is missing)."""


@dataclass(frozen=True)
class SchemaConfig:
    """Describes how a raw file maps onto our target schema."""

    column_map: dict  # raw column name -> target field name
    timestamp_format: str  # strptime format; timestamps are treated as naive UTC
    campaign_placeholders: tuple = ("-", "")  # raw values that mean "no campaign"
    conversion_true: tuple = ("yes", "1", "true")
    conversion_false: tuple = ("no", "0", "false")


# The Kaggle file: no event_id, no revenue, day-first timestamps, Yes/No conversion.
KAGGLE_CONFIG = SchemaConfig(
    column_map={
        "User ID": "user_id",
        "Timestamp": "timestamp",
        "Channel": "channel",
        "Campaign": "campaign",
        "Conversion": "conversion",
    },
    timestamp_format="%d-%m-%Y %H:%M",
)

# Our own target schema (used by tests and any future export that already follows it).
STANDARD_CONFIG = SchemaConfig(
    column_map={
        f: f
        for f in ("event_id", "user_id", "timestamp", "channel", "campaign", "conversion", "revenue")
    },
    timestamp_format="%Y-%m-%dT%H:%M:%SZ",
)


@dataclass
class ValidationSummary:
    rows_read: int
    rows_clean: int
    rows_rejected: int
    check_counts: dict  # check name -> rows failing it (one row can fail several checks)
    flagged_tie_rows: int
    flagged_possible_duplicate_rows: int
    campaign_placeholder_rows: int
    repeated_channel_touches: int
    n_users: int
    time_min: object
    time_max: object
    conversion_counts: dict
    channel_counts: dict
    event_id_derived: bool
    revenue_present: bool
    notes: list = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "=== VALIDATION SUMMARY ===",
            f"Rows read:      {self.rows_read}",
            f"Rows clean:     {self.rows_clean}",
            f"Rows rejected:  {self.rows_rejected}",
            f"Accounting check (read == clean + rejected): "
            f"{self.rows_read == self.rows_clean + self.rows_rejected}",
            "",
            "Checks (rows failing each; a row can fail more than one):",
        ]
        lines += [f"  {name}: {count}" for name, count in self.check_counts.items()]
        lines += [
            "",
            "Flagged but KEPT in clean data:",
            f"  rows sharing (user, timestamp): {self.flagged_tie_rows}",
            f"  rows with identical content at the same time (possible double-logging): "
            f"{self.flagged_possible_duplicate_rows}",
            "",
            "Informational (legitimate, preserved):",
            f"  repeated channel touches by the same user: {self.repeated_channel_touches}",
            f"  campaign placeholder rows mapped to '{NO_CAMPAIGN}': {self.campaign_placeholder_rows}",
            "",
            f"Users in clean data: {self.n_users}",
            f"Time range: {self.time_min} -> {self.time_max}",
            f"Conversion counts: {self.conversion_counts}",
            f"Channel counts: {self.channel_counts}",
            f"event_id derived from file row: {self.event_id_derived}",
            f"revenue column present: {self.revenue_present}",
        ]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    rejected: pd.DataFrame
    summary: ValidationSummary


# --------------------------------------------------------------------------- 1. load
def load_raw_csv(path) -> pd.DataFrame:
    """Read the CSV with every value kept as text, plus a 1-based `file_row` column.

    Reading everything as strings means pandas cannot silently turn a bad value into NaN or
    change a type: we do the interpretation ourselves, where we can report failures.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path.resolve()}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    df.insert(0, "file_row", range(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------- 2. columns
def validate_columns(raw: pd.DataFrame, config: SchemaConfig) -> list:
    """Raise SchemaError if a required column is missing.

    Returns the optional target fields that are absent from the file (e.g. revenue), so the
    summary can state plainly that they were not provided rather than invented.
    """
    present_targets = {t for r, t in config.column_map.items() if r in raw.columns}
    missing_raw = [r for r, t in config.column_map.items() if t in REQUIRED_FIELDS and r not in raw.columns]
    if missing_raw:
        raise SchemaError(f"Required column(s) missing from file: {missing_raw}. Found: {list(raw.columns)}")
    optional = ("event_id", "campaign", "revenue")
    return [f for f in optional if f not in present_targets]


def map_to_schema(raw: pd.DataFrame, config: SchemaConfig) -> pd.DataFrame:
    """Rename raw columns to target names, trim whitespace, and derive event_id if absent.

    Derived event_id = 'r' + file row zero-padded to 8 digits (r00000001). The padding makes
    text order equal file order, so the event_id tiebreaker is the original file order.
    """
    rename = {r: t for r, t in config.column_map.items() if r in raw.columns}
    df = raw.rename(columns=rename)[["file_row"] + list(rename.values())].copy()
    for col in df.columns.drop("file_row"):
        df[col] = df[col].str.strip()
    if "event_id" not in df.columns:
        df.insert(0, "event_id", "r" + df["file_row"].astype(str).str.zfill(8))
    return df


# --------------------------------------------------------------------------- 3. timestamps
def parse_timestamps(df: pd.DataFrame, fmt: str):
    """Parse strictly with `fmt`. Returns (parsed datetimes, mask of invalid rows).

    Invalid or blank timestamps become NaT here and are REJECTED (with a reason) by the caller.
    Parsed values are naive and assumed to be UTC.
    """
    parsed = pd.to_datetime(df["timestamp"], format=fmt, errors="coerce")
    return parsed, parsed.isna()


# --------------------------------------------------------------------------- 4. field checks
def check_user_ids(df: pd.DataFrame) -> pd.Series:
    """Mask of rows with an empty user_id (an event we cannot attach to any user)."""
    return df["user_id"] == ""


def check_channels(df: pd.DataFrame) -> dict:
    """Masks for an empty channel and for channels named like model states (Start/Conversion/Null)."""
    return {
        "missing_channel": df["channel"] == "",
        "reserved_channel_name": df["channel"].str.lower().isin(RESERVED_STATE_NAMES),
    }


def parse_conversion(df: pd.DataFrame, config: SchemaConfig):
    """Map Yes/No (or 1/0, true/false) to 1/0. Returns (Int64 flags, mask of invalid rows)."""
    lowered = df["conversion"].str.lower()
    flag = pd.Series(pd.NA, index=df.index, dtype="Int64")
    flag[lowered.isin(config.conversion_true)] = 1
    flag[lowered.isin(config.conversion_false)] = 0
    return flag, flag.isna()


def check_revenue(df: pd.DataFrame, conversion: pd.Series):
    """Validate revenue if the column exists. Returns (numeric revenue or None, dict of masks).

    Blank is allowed (revenue is optional). Rejected: non-numeric, non-finite, negative, or a
    positive amount on a non-conversion event.
    """
    if "revenue" not in df.columns:
        return None, {}
    nonblank = df["revenue"] != ""
    numeric = pd.to_numeric(df["revenue"], errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric.fillna(0))
    non_conversion = (conversion == 0).fillna(False).astype(bool)
    masks = {
        "invalid_revenue_not_numeric": nonblank & ~finite,
        "invalid_revenue_negative": finite & (numeric < 0),
        "invalid_revenue_on_non_conversion": finite & (numeric > 0) & non_conversion,
    }
    return numeric, masks


# --------------------------------------------------------------------------- 5. duplicate IDs
def handle_duplicate_event_ids(df: pd.DataFrame) -> dict:
    """Safe policy for repeated event_id values (blank ids are handled by their own check).

    * Exact duplicates (same id AND identical content): keep the FIRST, reject later copies.
      They are the same event logged twice, so counting both would inflate touches.
    * Conflicting duplicates (same id, different content): reject ALL rows in the group.
      We cannot know which version is right, and picking one would be a silent guess.
    """
    has_id = df["event_id"] != ""
    in_group = has_id & df.duplicated("event_id", keep=False)
    content_cols = [c for c in df.columns if c != "file_row"]
    distinct = df[in_group].drop_duplicates(subset=content_cols)
    sizes = distinct.groupby("event_id").size()
    conflict_ids = sizes[sizes > 1].index
    conflict = in_group & df["event_id"].isin(conflict_ids)
    exact_later = in_group & ~conflict & df.duplicated(subset=content_cols, keep="first")
    return {"duplicate_event_id_exact": exact_later, "duplicate_event_id_conflict": conflict}


# --------------------------------------------------------------------------- 6. sort
def sort_events(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by user_id, timestamp, event_id. Result is identical on every run.

    user_id is sorted as text ('10062' < '9'). Order *between* users does not matter for
    journeys; order *within* a user does, and that is fully determined by timestamp, then event_id.
    """
    return df.sort_values(["user_id", "timestamp", "event_id"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- 7. orchestrate
def clean_events(path=DEFAULT_RAW_PATH, config: SchemaConfig = KAGGLE_CONFIG) -> ValidationResult:
    raw = load_raw_csv(path)
    absent_optional = validate_columns(raw, config)
    mapped = map_to_schema(raw, config)

    parsed_ts, bad_ts = parse_timestamps(mapped, config.timestamp_format)
    conv_flag, bad_conv = parse_conversion(mapped, config)
    revenue, revenue_masks = check_revenue(mapped, conv_flag)

    checks = {
        "missing_event_id": mapped["event_id"] == "",
        "missing_user_id": check_user_ids(mapped),
        "invalid_timestamp": bad_ts,
        **check_channels(mapped),
        "invalid_conversion_flag": bad_conv,
        **revenue_masks,
        **handle_duplicate_event_ids(mapped),
    }

    reason_text = pd.Series("", index=mapped.index)
    for name, mask in checks.items():
        reason_text = reason_text.where(~mask, reason_text + name + ";")
    reason_text = reason_text.str.rstrip(";")
    reject = reason_text != ""

    # Rejected rows keep their ORIGINAL text values so you can see exactly what was wrong.
    rejected = mapped[reject].assign(reject_reasons=reason_text[reject]).reset_index(drop=True)

    clean = mapped.loc[~reject].copy()
    clean["timestamp"] = parsed_ts[~reject]
    clean["conversion"] = conv_flag[~reject].astype(int)
    if revenue is not None:
        clean["revenue"] = revenue[~reject]

    placeholder_rows = 0
    if "campaign" in clean.columns:
        is_placeholder = clean["campaign"].str.lower().isin([p.lower() for p in config.campaign_placeholders])
        placeholder_rows = int(is_placeholder.sum())
        clean.loc[is_placeholder, "campaign"] = NO_CAMPAIGN
    else:
        clean["campaign"] = NO_CAMPAIGN

    columns = ["event_id", "user_id", "timestamp", "channel", "campaign", "conversion"]
    if "revenue" in clean.columns:
        columns.append("revenue")
    clean = sort_events(clean[columns + ["file_row"]])

    # Flags: rows are KEPT. Ties are real events with ambiguous order; identical-content rows
    # may be double-logging, but we cannot prove it, so we flag instead of deleting.
    clean["tie_same_user_timestamp"] = clean.duplicated(["user_id", "timestamp"], keep=False)
    clean["possible_duplicate_content"] = clean.duplicated(
        ["user_id", "timestamp", "channel", "campaign", "conversion"], keep=False
    )

    notes = ["Timestamps are naive and assumed to be UTC (the source does not state a timezone)."]
    if "event_id" in absent_optional:
        notes.append("event_id was not in the file; derived from the original file row number.")
    if "revenue" in absent_optional:
        notes.append("revenue column absent: revenue was not validated and was not created.")
    if "campaign" in absent_optional:
        notes.append(f"campaign column absent: filled with '{NO_CAMPAIGN}'.")
    if len(rejected):
        notes.append(f"{len(rejected)} row(s) rejected. Inspect result.rejected before trusting results.")

    summary = ValidationSummary(
        rows_read=len(raw),
        rows_clean=len(clean),
        rows_rejected=len(rejected),
        check_counts={name: int(mask.sum()) for name, mask in checks.items()},
        flagged_tie_rows=int(clean["tie_same_user_timestamp"].sum()),
        flagged_possible_duplicate_rows=int(clean["possible_duplicate_content"].sum()),
        campaign_placeholder_rows=placeholder_rows,
        repeated_channel_touches=int(clean.duplicated(["user_id", "channel"], keep="first").sum()),
        n_users=int(clean["user_id"].nunique()),
        time_min=clean["timestamp"].min() if len(clean) else None,
        time_max=clean["timestamp"].max() if len(clean) else None,
        conversion_counts={int(k): int(v) for k, v in clean["conversion"].value_counts().sort_index().items()},
        channel_counts={str(k): int(v) for k, v in clean["channel"].value_counts().sort_index().items()},
        event_id_derived="event_id" in absent_optional,
        revenue_present=revenue is not None,
        notes=notes,
    )
    return ValidationResult(clean=clean, rejected=rejected, summary=summary)


def main() -> None:
    result = clean_events(DEFAULT_RAW_PATH, KAGGLE_CONFIG)
    print(result.summary.to_text())
    print("\nFirst 5 clean rows:")
    print(result.clean.head().to_string(index=False))
    if len(result.rejected):
        print("\nRejected rows (first 10):")
        print(result.rejected.head(10).to_string(index=False))
    else:
        print("\nNo rows rejected.")


if __name__ == "__main__":
    main()
