"""Run Stage 8 (Person B, Task 1.3): Shapley values, exported to data/processed/shapley_values.csv.

Run from the project root (after run_stage3.py, run_stage6.py and run_stage7.py):
    python scripts/run_stage8.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shapley_engine.shapley_engine import main  # noqa: E402

if __name__ == "__main__":
    main()
