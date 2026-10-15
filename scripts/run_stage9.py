#!/usr/bin/env python
"""Stage 9 - Task 1.4 Shapley validation runner.

Loads the REAL project outputs

    data/processed/coalition_values.csv   (Task 1.2)
    data/processed/shapley_values.csv     (Task 1.3)

runs the 13 validation checks and prints a report.  Exit code: 0 = all checks
passed, 1 = at least one check failed, 2 = BLOCKED (an upstream file is missing
or its columns cannot be interpreted).  It never invents data.

Usage:
    python scripts/run_stage9.py
    python scripts/run_stage9.py --baseline 0.0731 --tol 1e-9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shapley_engine.shapley_validation import (  # noqa: E402
    DEFAULT_TOL,
    SchemaError,
    validate_shapley_files,
)

DEFAULT_COALITION_VALUES = PROJECT_ROOT / "data" / "processed" / "coalition_values.csv"
DEFAULT_SHAPLEY_VALUES = PROJECT_ROOT / "data" / "processed" / "shapley_values.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "shapley_validation_report.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 1.4 - validate coalition values and Shapley values.")
    parser.add_argument("--coalition-values", type=Path, default=DEFAULT_COALITION_VALUES)
    parser.add_argument("--shapley-values", type=Path, default=DEFAULT_SHAPLEY_VALUES)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT, help="where to save the check table (CSV)")
    parser.add_argument("--no-report", action="store_true", help="do not write the report CSV")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="numerical tolerance for comparisons")
    parser.add_argument("--baseline", type=float, default=None, help="Person A baseline conversion probability; V(N) must match")
    parser.add_argument("--baseline-tol", type=float, default=1e-6)
    parser.add_argument("--channels", type=str, default=None, help="comma-separated expected channel list")
    parser.add_argument("--coalition-col", default=None)
    parser.add_argument("--value-col", default=None)
    parser.add_argument("--channel-col", default=None)
    parser.add_argument("--shapley-col", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = [c.strip() for c in args.channels.split(",") if c.strip()] if args.channels else None

    print("=" * 70)
    print("STAGE 9 / TASK 1.4 - SHAPLEY VALIDATION")
    print("=" * 70)
    print(f"Coalition values : {args.coalition_values}")
    print(f"Shapley values   : {args.shapley_values}")
    print()

    try:
        report = validate_shapley_files(
            args.coalition_values,
            args.shapley_values,
            expected_channels=expected,
            baseline=args.baseline,
            baseline_tol=args.baseline_tol,
            tol=args.tol,
            coalition_col=args.coalition_col,
            value_col=args.value_col,
            channel_col=args.channel_col,
            shapley_col=args.shapley_col,
        )
    except FileNotFoundError as exc:
        print("BLOCKED - required upstream output is missing.")
        print(f"FileNotFoundError: {exc}")
        print("REAL-DATA STATUS: BLOCKED")
        return 2
    except SchemaError as exc:
        print("BLOCKED - the upstream file's columns could not be interpreted.")
        print(f"SchemaError: {exc}")
        print("REAL-DATA STATUS: BLOCKED")
        return 2

    print(report.format())
    print()
    if not args.no_report:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        report.to_dataframe().to_csv(args.report_out, index=False)
        print(f"Check table saved to: {args.report_out}")
    using_defaults = args.coalition_values == DEFAULT_COALITION_VALUES and args.shapley_values == DEFAULT_SHAPLEY_VALUES
    if using_defaults:
        print("REAL-DATA STATUS: project files data/processed/coalition_values.csv and shapley_values.csv found and validated")
    else:
        print("INPUT STATUS: custom files validated - NOT the default data/processed project outputs, "
              "so do not report these as real project results unless they are")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
