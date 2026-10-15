"""Run Stage 6 (Person B, Task 1.1): players and coalitions, exported to data/processed/coalitions.csv.

Run from the project root (after run_stage3.py):  python scripts/run_stage6.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shapley_engine.coalition_generator import main  # noqa: E402

if __name__ == "__main__":
    main()

