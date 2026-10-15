"""Task 1.5 - Final Shapley integration (Person B).

Flow:   Task 1.2 coalition values + Task 1.3 Shapley values
            -> Task 1.4 validation  (gate: a failed validation BLOCKS integration)
            -> efficiency / channel-completeness re-check
            -> final channel-level Shapley attribution table (ready for Task 2)

What this module does NOT do: hybrid attribution (Task 2) and budget allocation
(Task 3).  It never changes a raw Shapley value.

Final table columns
-------------------
rank                    1 = largest Shapley value; ties broken alphabetically (deterministic)
channel                 marketing channel (Start / Conversion / Null are never channels)
shapley_value           RAW Shapley value exactly as produced by Task 1.3
normalized_weight       shapley_value / sum(shapley_value)   (kept separate from the raw value)
attribution_percentage  normalized_weight * 100   (an attribution weight, NOT a revenue share)

Normalization rule (explicit, documented behaviour)
---------------------------------------------------
* sum(phi) > tol and no negative phi  -> status OK: weights are in [0, 1] and sum to 1.
* sum(phi) > tol but some phi < -tol  -> status SIGNED_NEGATIVE_VALUES_PRESENT: the weights are the
  plain ratios phi_i / sum(phi).  They still sum to 1 but a negative phi gives a negative weight.
  Nothing is clipped, zeroed or replaced; Task 2 must decide how to treat negative weights.
* sum(phi) <= tol                     -> status UNDEFINED_NON_POSITIVE_TOTAL: dividing by a zero or
  negative total is not meaningful, so weights/percentages are left empty (NaN).

Raw Shapley values always satisfy sum(phi) = V(N) - V(empty), not sum(phi) = 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from shapley_engine.shapley_validation import (
    DEFAULT_TOL,
    PASS,
    RESERVED_STATES,
    ValidationReport,
    parse_coalition,
    validate_shapley_outputs,
)

FINAL_COLUMNS = ["rank", "channel", "shapley_value", "normalized_weight", "attribution_percentage"]

NORM_OK = "OK"
NORM_SIGNED = "SIGNED_NEGATIVE_VALUES_PRESENT"
NORM_UNDEFINED = "UNDEFINED_NON_POSITIVE_TOTAL"

INTERPRETATION_NOTE = (
    "shapley_value is the Shapley attribution of the Markov coalition-value game "
    "(V(S) = conversion probability with only channels in S retained). "
    "normalized_weight / attribution_percentage are attribution weights, NOT revenue shares; "
    "the dataset has no channel-level spend or revenue."
)


class IntegrationBlockedError(RuntimeError):
    """Integration was refused: Task 1.4 validation (or a re-check) failed.

    ``report`` holds the Task 1.4 ValidationReport when the block came from validation.
    """

    def __init__(self, message: str, report: Optional[ValidationReport] = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class IntegrationResult:
    """Everything produced by a successful integration."""

    table: pd.DataFrame
    meta: Dict[str, object]
    validation_report: ValidationReport
    output_path: Optional[Path]
    meta_path: Optional[Path]


# --------------------------------------------------------------------------- #
# Normalization and ranking
# --------------------------------------------------------------------------- #
def normalize_shapley(phi: Mapping[str, float], tol: float = DEFAULT_TOL) -> Tuple[Optional[Dict[str, float]], str]:
    """Return ({channel: phi / sum(phi)} or None, status).  See the module docstring."""
    total = math.fsum(phi.values())
    if total <= tol:
        return None, NORM_UNDEFINED
    weights = {channel: value / total for channel, value in phi.items()}
    status = NORM_SIGNED if any(value < -tol for value in phi.values()) else NORM_OK
    return weights, status


def rank_channels(phi: Mapping[str, float]) -> Dict[str, int]:
    """Deterministic ranks 1..N: larger phi first; equal values (to 12 decimals) sorted by name."""
    order = sorted(phi, key=lambda c: (-round(phi[c], 12), c.casefold(), c))
    return {channel: position for position, channel in enumerate(order, start=1)}


def build_final_table(phi: Mapping[str, float], tol: float = DEFAULT_TOL) -> Tuple[pd.DataFrame, str]:
    """Build the final attribution table (sorted by rank) and return it with the normalization status."""
    ranks = rank_channels(phi)
    weights, status = normalize_shapley(phi, tol)
    rows = []
    for channel in sorted(phi, key=lambda c: ranks[c]):
        weight = weights[channel] if weights is not None else float("nan")
        rows.append(
            {
                "rank": ranks[channel],
                "channel": channel,
                "shapley_value": float(phi[channel]),
                "normalized_weight": weight,
                "attribution_percentage": weight * 100.0,
            }
        )
    return pd.DataFrame(rows, columns=FINAL_COLUMNS), status


# --------------------------------------------------------------------------- #
# Task 2 input contract
# --------------------------------------------------------------------------- #
def check_task2_ready(table: pd.DataFrame, tol: float = 1e-9) -> Tuple[List[str], List[str]]:
    """Check the final table against what Task 2 (hybrid attribution) needs.

    Returns (problems, warnings).  Task 2 is ready only if there are no problems.
    """
    problems: List[str] = []
    warnings: List[str] = []
    missing_cols = [c for c in FINAL_COLUMNS if c not in table.columns]
    if missing_cols:
        return [f"missing column(s): {', '.join(missing_cols)}"], warnings
    if table.empty:
        return ["table has no channels"], warnings

    channels = table["channel"].astype(str).str.strip()
    if (channels == "").any() or table["channel"].isna().any():
        problems.append("blank channel name")
    if channels.duplicated().any():
        problems.append("duplicate channel(s): " + ", ".join(sorted(set(channels[channels.duplicated()]))))
    reserved = sorted({c for c in channels if c.lower() in RESERVED_STATES})
    if reserved:
        problems.append("reserved state(s) present as channel: " + ", ".join(reserved))

    phi = pd.to_numeric(table["shapley_value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(phi).all():
        problems.append("shapley_value contains non-numeric, NaN or infinite values")

    ranks = sorted(pd.to_numeric(table["rank"], errors="coerce").tolist())
    if ranks != list(range(1, len(table) + 1)):
        problems.append("rank is not exactly 1..N")

    weights = pd.to_numeric(table["normalized_weight"], errors="coerce").to_numpy(dtype=float)
    pct = pd.to_numeric(table["attribution_percentage"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(weights).all():
        problems.append("normalized_weight is undefined (sum of Shapley values is not positive)")
    elif not np.isfinite(weights).all():
        problems.append("normalized_weight has NaN/infinite entries")
    else:
        if not math.isclose(float(weights.sum()), 1.0, rel_tol=tol, abs_tol=tol):
            problems.append(f"normalized_weight sums to {weights.sum():.12g}, expected 1")
        if not np.allclose(pct, weights * 100.0, rtol=tol, atol=tol):
            problems.append("attribution_percentage is not normalized_weight * 100")
        if (weights < -tol).any():
            warnings.append("negative normalized_weight present (a channel has negative Shapley value); "
                            "Task 2 must decide how to treat it")
    return problems, warnings


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
def _grand_and_empty_values(coalition_df: pd.DataFrame, report: ValidationReport) -> Tuple[float, float]:
    """V(N) and V(empty) read from the (already validated) coalition table."""
    ccol, vcol = report.columns_used["coalition"], report.columns_used["value"]
    values = {parse_coalition(cell): float(v) for cell, v in zip(coalition_df[ccol], coalition_df[vcol])}
    return values[frozenset(report.channels)], values[frozenset()]


def integrate_shapley_frames(
    coalition_df: pd.DataFrame,
    shapley_df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    tol: float = DEFAULT_TOL,
    source_files: Optional[Mapping[str, str]] = None,
    **validation_kwargs: object,
) -> IntegrationResult:
    """Validate (Task 1.4) and, only if validation passes, build/save the final attribution.

    ``validation_kwargs`` are forwarded to ``validate_shapley_outputs`` (expected_channels,
    baseline, baseline_tol, column-name overrides, shapley_fn).  Validation cannot be skipped.
    Raises ``IntegrationBlockedError`` (and writes nothing) if anything fails.
    """
    report = validate_shapley_outputs(coalition_df, shapley_df, tol=tol, **validation_kwargs)
    if not report.passed:
        failed = [f"{r.number} {r.name}" for r in report.results if r.status != PASS]
        raise IntegrationBlockedError("Task 1.4 validation failed: " + "; ".join(failed), report)

    cols = report.columns_used
    labels = [str(x).strip() for x in shapley_df[cols["channel"]]]
    # float() parses exactly (pandas' string parser can be 1 ulp off), so raw values stay untouched.
    phi = {label: float(cell.strip() if isinstance(cell, str) else cell)
           for label, cell in zip(labels, shapley_df[cols["shapley"]].astype(object))}

    # Re-checks (defence in depth; validation already covers them).
    if set(phi) != set(report.channels) or len(phi) != len(labels):
        raise IntegrationBlockedError("Final channels do not match the channels of the Shapley game", report)
    reserved = sorted(c for c in phi if c.lower() in RESERVED_STATES)
    if reserved:
        raise IntegrationBlockedError("Start/Conversion/Null cannot be channels: " + ", ".join(reserved), report)
    v_grand, v_empty = _grand_and_empty_values(coalition_df, report)
    total = math.fsum(phi.values())
    expected = v_grand - v_empty
    if not math.isclose(total, expected, rel_tol=tol, abs_tol=tol):
        raise IntegrationBlockedError(
            f"Efficiency failed: sum(phi)={total:.12g} but V(N)-V(empty)={expected:.12g}. "
            "The Shapley values were not modified; check Task 1.2/1.3 outputs.",
            report,
        )

    table, norm_status = build_final_table(phi, tol)
    problems, warnings = check_task2_ready(table, tol)
    meta: Dict[str, object] = {
        "task": "1.5 final Shapley integration",
        "n_channels": len(phi),
        "channels": list(table["channel"]),
        "v_empty": v_empty,
        "v_grand": v_grand,
        "total_shapley": total,
        "expected_total": expected,
        "efficiency_abs_diff": abs(total - expected),
        "efficiency_passed": True,
        "tolerance": tol,
        "normalization_status": norm_status,
        "has_negative_shapley_values": bool((table["shapley_value"] < -tol).any()),
        "task2_ready": not problems,
        "task2_problems": problems,
        "task2_warnings": warnings,
        "validation": {
            "passed": True,
            "n_checks": len(report.results),
            "checks": [{"number": r.number, "name": r.name, "status": r.status} for r in report.results],
        },
        "interpretation": INTERPRETATION_NOTE,
    }
    if source_files:
        meta["source_files"] = dict(source_files)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False, lineterminator="\n")
    if meta_path is not None:
        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return IntegrationResult(table, meta, report, output_path, meta_path)


def _load_csv(path: Path, description: str, producer_hint: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found ({description}). {producer_hint}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def integrate_shapley_files(
    coalition_values_path: Path,
    shapley_values_path: Path,
    output_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    **kwargs: object,
) -> IntegrationResult:
    """Load the Task 1.2 / Task 1.3 CSVs and run :func:`integrate_shapley_frames`."""
    coalition_df = _load_csv(
        coalition_values_path,
        "Task 1.2 coalition values",
        "Run the Task 1.2 stage that produces it first (e.g. python scripts/run_stage7.py).",
    )
    shapley_df = _load_csv(
        shapley_values_path,
        "Task 1.3 Shapley values",
        "Run the Task 1.3 stage that produces it first (e.g. python scripts/run_stage8.py).",
    )
    sources = {
        "coalition_values": Path(coalition_values_path).name,
        "shapley_values": Path(shapley_values_path).name,
    }
    return integrate_shapley_frames(
        coalition_df, shapley_df, output_path=output_path, meta_path=meta_path, source_files=sources, **kwargs
    )
