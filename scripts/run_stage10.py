#!/usr/bin/env python
"""Stage 10 - Task 1.5 final Shapley integration runner.

Loads the REAL project outputs

    data/processed/coalition_values.csv   (Task 1.2)
    data/processed/shapley_values.csv     (Task 1.3)

runs Task 1.4 validation, and only if it passes writes

    data/processed/final_shapley_attribution.csv
    data/processed/final_shapley_attribution_meta.json

Exit codes: 0 = integration succeeded, 1 = blocked by validation/efficiency failure,
2 = BLOCKED because an upstream file is missing or its columns cannot be interpreted.
It never invents data and never skips validation.

Usage:
    python scripts/run_stage10.py
    python scripts/run_stage10.py --baseline 0.0731
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shapley_engine.shapley_integration import (  # noqa: E402
    IntegrationBlockedError,
    integrate_shapley_files,
)
from shapley_engine.shapley_validation import DEFAULT_TOL, PASS, SchemaError  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_COALITION_VALUES = PROCESSED / "coalition_values.csv"
DEFAULT_SHAPLEY_VALUES = PROCESSED / "shapley_values.csv"
DEFAULT_OUTPUT = PROCESSED / "final_shapley_attribution.csv"
DEFAULT_META = PROCESSED / "final_shapley_attribution_meta.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 1.5 - final Shapley integration (validation-gated).")
    parser.add_argument("--coalition-values", type=Path, default=DEFAULT_COALITION_VALUES)
    parser.add_argument("--shapley-values", type=Path, default=DEFAULT_SHAPLEY_VALUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
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

    print("=== Final Shapley Integration ===")
    print()
    print(f"Coalition values : {args.coalition_values}")
    print(f"Shapley values   : {args.shapley_values}")
    print()

    try:
        result = integrate_shapley_files(
            args.coalition_values,
            args.shapley_values,
            output_path=args.output,
            meta_path=args.meta_out,
            tol=args.tol,
            expected_channels=expected,
            baseline=args.baseline,
            baseline_tol=args.baseline_tol,
            coalition_col=args.coalition_col,
            value_col=args.value_col,
            channel_col=args.channel_col,
            shapley_col=args.shapley_col,
        )
    except FileNotFoundError as exc:
        print("BLOCKED - required upstream output is missing.")
        print(f"FileNotFoundError: {exc}")
        print("Integration: NOT RUN")
        print("REAL-DATA STATUS: BLOCKED")
        return 2
    except SchemaError as exc:
        print("BLOCKED - the upstream file's columns could not be interpreted.")
        print(f"SchemaError: {exc}")
        print("Integration: NOT RUN")
        print("REAL-DATA STATUS: BLOCKED")
        return 2
    except IntegrationBlockedError as exc:
        print(f"Validation: FAILED")
        print(f"Integration: BLOCKED by validation failure - {exc}")
        if exc.report is not None:
            print()
            for r in exc.report.results:
                if r.status != PASS:
                    print(f"[{r.status}] {r.number:>2} {r.name}: {r.detail}")
        print()
        print("No final output was written by this run.")
        if args.output.exists():
            print(f"WARNING: {args.output} exists from an earlier run and may be stale.")
        return 1

    meta, table = result.meta, result.table
    counts = result.validation_report.counts()
    print(f"Channels: {meta['n_channels']}")
    print(f"Validation: PASSED ({counts[PASS]}/{len(result.validation_report.results)} checks)")
    print("Efficiency: PASSED")
    print()
    print("Final Shapley Attribution:")
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()
    print(f"Total Shapley: {meta['total_shapley']:.10f}")
    print(f"Expected (V(N) - V(empty)): {meta['expected_total']:.10f}   [V(N)={meta['v_grand']:.10f}, V(empty)={meta['v_empty']:.10f}]")
    print(f"Normalization: {meta['normalization_status']}")
    print(f"Task 2 ready: {'yes' if meta['task2_ready'] else 'NO - ' + '; '.join(meta['task2_problems'])}")
    for warning in meta["task2_warnings"]:
        print(f"Warning: {warning}")
    print("Note: weights/percentages are attribution weights, not revenue shares.")
    print()
    print(f"Saved: {result.output_path}")
    print(f"Saved: {result.meta_path}")
    print()
    print("Integration: SUCCESS")
    using_defaults = args.coalition_values == DEFAULT_COALITION_VALUES and args.shapley_values == DEFAULT_SHAPLEY_VALUES
    if using_defaults:
        print("REAL-DATA STATUS: project files data/processed/coalition_values.csv and shapley_values.csv processed")
    else:
        print("INPUT STATUS: custom files processed - NOT the default data/processed project outputs, "
              "so do not report these as real project results unless they are")
    return 0


if __name__ == "__main__":
    sys.exit(main())
