"""Stage 11: hybrid attribution = alpha * Markov weight + (1 - alpha) * Shapley weight.

Scope: combine the two EXISTING attribution outputs by channel name. This module does not
recompute Markov or Shapley results, does not allocate budget, and has no UI.

Inputs (read from CSV, schemas verified against the project's real outputs)
  markov_results.csv            segment, channel, baseline_conversion_probability,
                                conversion_probability_without_channel, removal_effect, status, ...
  final_shapley_attribution.csv rank, channel, shapley_value, normalized_weight, attribution_percentage

Normalisation policies (stated once, here)
  Markov : weight(k) = RE(k) / sum_j RE(j). Removal effects each measure a SEPARATE removal and sum
           to more than 1, so they are not shares until divided by their total. Requires every
           effect to be finite and non-negative and the total to be > 0, otherwise we raise.
  Shapley: raw values are kept signed and untouched. Allocation weights use a POSITIVE-PART policy:
           negatives are clipped to 0 in a separate step, the rest are divided by their sum, and
           clipped channels are flagged. If everything clips to 0 we raise. When all raw values are
           positive this equals raw / sum, which is what the existing normalized_weight column
           contains (checked numerically by `cross_checks`).
  Clipping discards HOW negative a channel was, so a clipped channel gets weight 0 in the hybrid.

Hybrid: both weight vectors are non-negative and sum to 1, so alpha*M + (1-alpha)*S is non-negative
and sums to 1 exactly (up to float rounding). No extra renormalisation is applied; the sum is
verified and an error is raised if it ever fails.

Consistency guard: with redirect-to-Null, Shapley values must sum to the grand-coalition value, the
Markov baseline conversion probability, if both methods used the same journeys. `cross_checks`
compares the two and, by default, hybrid_attribution refuses to mix outputs that disagree. The
Shapley file has no segment column, so only the 'all' segment can pass this check.

Limitations: the weights are attribution scores, not causal lift, revenue or ROI; and on this
dataset the channel scores are within bootstrap noise of one another (see the Stage 5 notes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MARKOV_PATH = Path("data/processed/markov_results.csv")
DEFAULT_SHAPLEY_PATH = Path("data/processed/final_shapley_attribution.csv")  # location assumed; confirm
DEFAULT_HYBRID_PATH = Path("data/processed/hybrid_attribution.csv")
RESERVED_STATES = {"start", "conversion", "null"}
ZERO_TOL = 1e-12        # a total at or below this cannot be normalised
SUM_TOL = 1e-9          # tolerance for "weights sum to 1"
CONSISTENCY_TOL = 1e-5  # the Shapley CSV is stored to 6 decimals, so exact equality is not expected
HYBRID_COLUMNS = [
    "segment", "alpha", "channel", "markov_removal_effect", "markov_weight", "shapley_raw_value",
    "shapley_clipped", "shapley_weight", "hybrid_score", "hybrid_weight", "hybrid_percentage",
]


# --------------------------------------------------------------------------- loading
def _require_columns(df: pd.DataFrame, columns, name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s) {missing}; found {list(df.columns)}")


def load_markov_results(path=DEFAULT_MARKOV_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Generate it with `python scripts/run_stage5.py`.")
    df = pd.read_csv(path, keep_default_na=False)  # blank diagnostic cells stay as '' not NaN
    _require_columns(df, ["segment", "channel", "removal_effect", "status",
                          "baseline_conversion_probability"], "Markov results")
    return df


def load_shapley_results(path=DEFAULT_SHAPLEY_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Pass the real location of final_shapley_attribution.csv "
            "(DEFAULT_SHAPLEY_PATH is an assumption)."
        )
    df = pd.read_csv(path, keep_default_na=False)
    _require_columns(df, ["channel", "shapley_value"], "Shapley results")
    return df


# --------------------------------------------------------------------------- validation helpers
def validate_alpha(alpha) -> float:
    """alpha must be a real number in [0, 1]. Booleans and strings are rejected, not coerced."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0 or a > 1.0:
        raise ValueError(f"alpha must be finite and between 0 and 1 inclusive, got {alpha!r}")
    return a


def _check_channels(df: pd.DataFrame, name: str) -> None:
    ch = df["channel"].astype(str)
    if (ch.str.strip() == "").any():
        raise ValueError(f"{name}: blank channel name")
    reserved = sorted(set(ch[ch.str.lower().isin(RESERVED_STATES)]))
    if reserved:
        raise ValueError(f"{name}: model state(s) {reserved} appear as marketing channels")
    dup = sorted(set(ch[ch.duplicated()]))
    if dup:
        raise ValueError(f"{name}: duplicate channel row(s) for {dup}; one row per channel is required")


def _finite_numbers(df: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce").astype(float)
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(f"{name}: missing or non-finite {column!r} for channel(s) {sorted(df.loc[bad, 'channel'])}")
    return values


# --------------------------------------------------------------------------- Markov side
def select_markov_segment(markov: pd.DataFrame, segment: str = "all") -> pd.DataFrame:
    """Rows for one segment; every row must have status 'ok' (nothing is silently dropped)."""
    available = sorted(markov["segment"].astype(str).unique())
    if segment not in available:
        raise ValueError(f"Markov segment {segment!r} not available; found {available}")
    sel = markov[markov["segment"].astype(str) == segment].copy()
    _check_channels(sel, f"Markov[{segment}]")
    not_ok = sel[sel["status"] != "ok"]
    if len(not_ok):
        detail = dict(zip(not_ok["channel"], not_ok["status"]))
        raise ValueError(
            f"Markov[{segment}]: channel(s) with status other than 'ok' {detail}. These cannot be "
            "hybridised; fix the Markov run (e.g. a zero baseline) instead of dropping them."
        )
    sel["removal_effect"] = _finite_numbers(sel, "removal_effect", f"Markov[{segment}]")
    sel["baseline_conversion_probability"] = _finite_numbers(
        sel, "baseline_conversion_probability", f"Markov[{segment}]")
    if not np.allclose(sel["baseline_conversion_probability"], sel["baseline_conversion_probability"].iloc[0]):
        raise ValueError(f"Markov[{segment}]: channels use different baselines; expected one shared baseline")
    return sel.reset_index(drop=True)


def markov_weights(removal_effects: pd.Series) -> pd.Series:
    """RE(k) / sum(RE). Raises on negative effects or a zero total."""
    if (removal_effects < 0).any():
        raise ValueError("removal effects must be non-negative to form weights")
    total = float(removal_effects.sum())
    if total <= ZERO_TOL:
        raise ValueError("sum of removal effects is zero, so Markov weights are undefined")
    return removal_effects / total


# --------------------------------------------------------------------------- Shapley side
def prepare_shapley(shapley: pd.DataFrame) -> pd.DataFrame:
    _check_channels(shapley, "Shapley")
    out = shapley.copy()
    out["shapley_value"] = _finite_numbers(out, "shapley_value", "Shapley")
    return out.reset_index(drop=True)


def shapley_weights(raw: pd.Series):
    """Positive-part normalisation. Returns (weights, clipped_flags); raw values are not modified."""
    clipped = raw < 0
    positive = raw.clip(lower=0.0)
    total = float(positive.sum())
    if total <= ZERO_TOL:
        raise ValueError("all Shapley values are zero or negative, so no valid allocation weights exist")
    return positive / total, clipped


# --------------------------------------------------------------------------- consistency
def cross_checks(markov_sel: pd.DataFrame, shapley: pd.DataFrame, tol: float = CONSISTENCY_TOL) -> dict:
    """Evidence about whether both methods describe the same journeys.

    baseline_difference: |sum of Shapley raw values - Markov baseline|. Under redirect-to-Null the
    two must match (Shapley efficiency). A large gap means different journeys or definitions.
    existing_weight_max_diff: how far the Shapley file's own normalized_weight is from raw/sum.
    """
    baseline = float(markov_sel["baseline_conversion_probability"].iloc[0])
    total = float(shapley["shapley_value"].sum())
    result = {
        "markov_baseline": baseline,
        "shapley_sum": total,
        "baseline_difference": abs(total - baseline),
        "baseline_consistent": abs(total - baseline) <= tol,
        "existing_weight_max_diff": None,
        "existing_weights_match": None,
    }
    if "normalized_weight" in shapley.columns and abs(total) > ZERO_TOL:
        existing = pd.to_numeric(shapley["normalized_weight"], errors="coerce")
        diff = float(np.abs(existing - shapley["shapley_value"] / total).max())
        result["existing_weight_max_diff"] = diff
        result["existing_weights_match"] = bool(diff <= tol)
    return result


# --------------------------------------------------------------------------- the hybrid
def hybrid_attribution(markov: pd.DataFrame, shapley: pd.DataFrame, alpha: float = 0.5,
                       segment: str = "all", check_consistency: bool = True) -> pd.DataFrame:
    """Channel-level hybrid table for one Markov segment (see HYBRID_COLUMNS).

    Channels are matched by name, never by row position or rank. Any mismatch, duplicate,
    reserved state name, missing/non-finite score, invalid status or zero-sum distribution raises.
    """
    a = validate_alpha(alpha)
    m = select_markov_segment(markov, segment)
    s = prepare_shapley(shapley)

    only_m = sorted(set(m["channel"]) - set(s["channel"]))
    only_s = sorted(set(s["channel"]) - set(m["channel"]))
    if only_m or only_s:
        raise ValueError(f"channel sets differ. Only in Markov: {only_m}. Only in Shapley: {only_s}.")

    if check_consistency:
        checks = cross_checks(m, s)
        if not checks["baseline_consistent"]:
            raise ValueError(
                f"Shapley values sum to {checks['shapley_sum']:.6f} but the Markov[{segment}] baseline is "
                f"{checks['markov_baseline']:.6f} (difference {checks['baseline_difference']:.2e}). The two "
                "runs may use different journeys, definitions or segments. Fix the inputs, or pass "
                "check_consistency=False if you have verified they are compatible."
            )

    table = m[["channel", "removal_effect"]].rename(columns={"removal_effect": "markov_removal_effect"}).merge(
        s[["channel", "shapley_value"]].rename(columns={"shapley_value": "shapley_raw_value"}),
        on="channel", how="inner", validate="one_to_one")
    table["markov_weight"] = markov_weights(table["markov_removal_effect"])
    table["shapley_weight"], table["shapley_clipped"] = shapley_weights(table["shapley_raw_value"])
    table["hybrid_score"] = a * table["markov_weight"] + (1.0 - a) * table["shapley_weight"]
    table["hybrid_weight"] = table["hybrid_score"]  # already a distribution; see module docstring
    table["hybrid_percentage"] = 100.0 * table["hybrid_weight"]
    table.insert(0, "alpha", a)
    table.insert(0, "segment", segment)

    if (table["hybrid_weight"] < -SUM_TOL).any() or abs(float(table["hybrid_weight"].sum()) - 1.0) > SUM_TOL:
        raise ValueError("hybrid weights are not a valid distribution; check the input weights")
    table["_key"] = table["hybrid_weight"].round(12)  # rounding keeps tie order identical on every machine
    table = table.sort_values(["_key", "channel"], ascending=[False, True]).drop(columns="_key")
    return table[HYBRID_COLUMNS].reset_index(drop=True)


def export_hybrid(table: pd.DataFrame, path=DEFAULT_HYBRID_PATH) -> Path:
    """Write the hybrid table (full float precision), creating the folder. Overwrites the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 11: hybrid Markov + Shapley attribution")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight on Markov, 0..1 (default 0.5)")
    parser.add_argument("--segment", default="all", help="Markov segment (default all)")
    parser.add_argument("--markov", default=str(DEFAULT_MARKOV_PATH))
    parser.add_argument("--shapley", default=str(DEFAULT_SHAPLEY_PATH))
    parser.add_argument("--out", default=str(DEFAULT_HYBRID_PATH))
    args = parser.parse_args(argv)

    markov = load_markov_results(args.markov)
    shapley = load_shapley_results(args.shapley)
    checks = cross_checks(select_markov_segment(markov, args.segment), prepare_shapley(shapley))
    print("=== CONSISTENCY CHECKS ===")
    print(f"Markov baseline ({args.segment}):      {checks['markov_baseline']:.6f}")
    print(f"Sum of raw Shapley values:        {checks['shapley_sum']:.6f}")
    print(f"Difference:                       {checks['baseline_difference']:.2e}  consistent={checks['baseline_consistent']}")
    print(f"Existing Shapley normalized_weight vs raw/sum, max diff: {checks['existing_weight_max_diff']}  "
          f"match={checks['existing_weights_match']}")

    table = hybrid_attribution(markov, shapley, alpha=args.alpha, segment=args.segment)
    print(f"\n=== HYBRID ATTRIBUTION (segment={args.segment}, alpha={args.alpha}) ===")
    print(table.drop(columns=["segment", "alpha"]).round(6).to_string(index=False))
    print(f"\nSum of hybrid weights: {table['hybrid_weight'].sum():.12f}   "
          f"clipped Shapley channels: {int(table['shapley_clipped'].sum())}")
    out = export_hybrid(table, args.out)
    print(f"Exported {len(table)} rows to {out}")


if __name__ == "__main__":
    main()
