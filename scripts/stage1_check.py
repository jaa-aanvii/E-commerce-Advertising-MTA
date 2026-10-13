"""Stage 1 inspection of the Kaggle MTA file. Read-only: prints facts, changes nothing."""
from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/raw/multi_touch_attribution_data.csv")
TS_FORMAT = "%d-%m-%Y %H:%M"  # day-first, e.g. 10-02-2025 07:58


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values per column:")
    print(df.isna().sum().to_string())

    ts = pd.to_datetime(df["Timestamp"], format=TS_FORMAT, errors="coerce")
    print("Unparseable timestamps:", int(ts.isna().sum()))
    print("Time range:", ts.min(), "->", ts.max())
    print("File already in chronological order:", ts.is_monotonic_increasing)

    print("Unique users:", df["User ID"].nunique())
    print("Conversion values:", df["Conversion"].value_counts().to_dict())
    print("Channels:", df["Channel"].value_counts().to_dict())
    print("Campaigns:", df["Campaign"].value_counts().to_dict())

    tied = df.assign(ts=ts).duplicated(subset=["User ID", "ts"], keep=False)
    print("Rows sharing the same (user, timestamp):", int(tied.sum()))
    print("Fully duplicated rows:", int(df.duplicated().sum()))

    print("\nInspect two users by hand (sorted by time):")
    show = df.assign(ts=ts, file_row=range(1, len(df) + 1))
    for uid in (10062, 10167):
        print(show[show["User ID"] == uid].sort_values(["ts", "file_row"])
              .drop(columns="ts").to_string(index=False))


if __name__ == "__main__":
    main()