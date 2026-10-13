"""Run Stage 5: channel removal effects, exported to data/processed/markov_results.csv.

Run from the project root (after run_stage3.py):  python scripts/run_stage5.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mta.markov_engine import main_removal  # noqa: E402

if __name__ == "__main__":
    main_removal()
