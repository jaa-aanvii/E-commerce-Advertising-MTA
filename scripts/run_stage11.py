"""Run Stage 11: hybrid Markov + Shapley attribution, exported to data/processed/hybrid_attribution.csv.

Run from the project root:  python scripts/run_stage11.py            (alpha 0.5, segment all)
Options:                    python scripts/run_stage11.py --alpha 0.7 --shapley path\to\final_shapley_attribution.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid_engine import main  # noqa: E402

if __name__ == "__main__":
    main()
