"""Run Stage 3: build journeys from the raw Kaggle file and export data/processed/journeys.csv.

Run from the project root:  python scripts/run_stage3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mta.journey_builder import main  # noqa: E402

if __name__ == "__main__":
    main()
