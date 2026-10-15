"""Run Stage 7 (Person B, Task 1.2): coalition values V(S), exported to data/processed/coalition_values.csv.

Run from the project root (after run_stage3.py and run_stage6.py):  python scripts/run_stage7.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coalition_value import main  # noqa: E402

if __name__ == "__main__":
    main()