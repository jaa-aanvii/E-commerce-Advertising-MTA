"""Run Stage 12: allocate a hypothetical budget by hybrid attribution weight.

Run from the project root (after run_stage11.py):
    python scripts/run_stage12.py                          (budget Rs 10,00,000)
    python scripts/run_stage12.py --budget 25,00,000
    python scripts/run_stage12.py --budget 500000 --out data/processed/budget_allocation.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from budget_allocator import main  # noqa: E402

if __name__ == "__main__":
    main()
