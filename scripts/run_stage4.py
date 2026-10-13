"""Run Stage 4: fit the Markov transition model on data/processed/journeys.csv.

Run from the project root (after run_stage3.py):  python scripts/run_stage4.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mta.markov_engine import main  # noqa: E402

if __name__ == "__main__":
    main()
