"""Stage 12: contribution-based budget allocation from hybrid attribution weights.

Scope: turn channel weights plus a user-entered total budget into rupee amounts. No dashboard,
no revenue model, no ROI.

Formula:   Recommended Budget(k) = Hybrid Weight(k) x Total Budget

INTERPRETATION (read before presenting any number)
  This is a HYPOTHETICAL, CONTRIBUTION-BASED split. Each weight comes from historical journeys.
  The dataset has no channel-level spend or revenue, so nothing here measures ROI, ROAS, profit or
  incremental revenue. A larger weight does NOT mean a channel is more cost-effective, and nothing
  shows the split is optimal or will produce any particular return. Attribution weights are model
  scores, not causal impact, so small differences between channels are not evidence that one
  channel is better. See ALLOCATION_NOTICE, which the dashboard should display.

Whole-rupee rounding (deterministic, always reconciles)
  Amounts are whole rupees, so the total budget must itself be a whole number of rupees (a
  fractional budget is rejected, never silently rounded). Each channel's exact share is computed
  with exact fractions, floored to whole rupees, and the leftover rupees (always fewer than the
  number of channels) go one each to the channels with the largest fractional parts. Ties are
  broken alphabetically by channel name. Result: amounts add up to the budget EXACTLY, each is
  within one rupee of its exact share, a zero-weight channel never receives a leftover rupee, and
  the output does not depend on input row order.

Validation (all raise, nothing is silently corrected)
  budget  : a real number (not bool/str/None), finite, whole rupees, 0 <= budget <= MAX_BUDGET.
  weights : one row per channel, no blanks/duplicates/Start-Conversion-Null, finite, non-negative,
            summing to 1 within WEIGHT_TOL. Because of that tolerance the exact shares are taken
            from weight / sum(weights), which makes the amounts reconcile exactly.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HYBRID_PATH = Path("data/processed/hybrid_attribution.csv")
RESERVED_STATES = {"start", "conversion", "null"}
WEIGHT_TOL = 1e-6        # tolerance for "weights sum to 1" (a 6-decimal CSV is off by about 1e-6)
MAX_BUDGET = 10 ** 15    # keeps floats exact (below 2**53) and rejects absurd inputs
ALLOCATION_COLUMNS = ["channel", "hybrid_weight", "hybrid_percentage", "recommended_budget",
                      "unrounded_budget", "rounding_adjustment"]

ALLOCATION_NOTICE = (
    "This is a hypothetical, contribution-based allocation: each channel's share is its hybrid "
    "attribution weight from historical journeys, multiplied by the budget you entered.",
    "The dataset has no channel-level spend or revenue, so this does not calculate ROI, ROAS, "
    "profit or incremental revenue.",
    "A larger weight does not mean a channel is more cost-effective, and nothing here shows the "
    "split is optimal or will produce any particular return.",
    "Attribution weights are model-based scores, not causal impact. Small differences between "
    "channels should not be read as one channel being better.",
)


# --------------------------------------------------------------------------- budget input
def parse_budget_text(text) -> "int | float":
    """'10,00,000', '₹ 10,00,000' and '1000000' all read as 1000000. For typed input boxes."""
    cleaned = str(text).replace("₹", "").replace(",", "").replace(" ", "").strip()
    if cleaned == "":
        raise ValueError("budget is empty")
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        raise ValueError(f"cannot read {text!r} as a number") from None


def validate_budget(budget) -> int:
    """Return the budget as a whole number of rupees, or raise."""
    if isinstance(budget, bool) or not isinstance(budget, (int, float, np.integer, np.floating)):
        raise TypeError(f"budget must be a number, got {type(budget).__name__}")
    if isinstance(budget, (float, np.floating)):
        if not np.isfinite(budget):
            raise ValueError(f"budget must be finite, got {budget!r}")
        if not float(budget).is_integer():
            raise ValueError(f"budget must be a whole number of rupees, got {budget!r}")
    value = int(budget)
    if value < 0:
        raise ValueError(f"budget must be non-negative, got {value}")
    if value > MAX_BUDGET:
        raise ValueError(f"budget {value} exceeds the supported maximum of {MAX_BUDGET}")
    return value


# --------------------------------------------------------------------------- weights input
def validate_weights(weights: pd.DataFrame, tol: float = WEIGHT_TOL) -> pd.DataFrame:
    """Check a channel/hybrid_weight table and return a clean copy (channel, hybrid_weight)."""
    missing = [c for c in ("channel", "hybrid_weight") if c not in weights.columns]
    if missing:
        raise ValueError(f"weights table is missing required column(s) {missing}; found {list(weights.columns)}")
    if weights.empty:
        raise ValueError("weights table has no channels")
    channel = weights["channel"].astype(str)
    if (channel.str.strip() == "").any():
        raise ValueError("blank channel name in weights table")
    reserved = sorted(set(channel[channel.str.lower().isin(RESERVED_STATES)]))
    if reserved:
        raise ValueError(f"model state(s) {reserved} appear as marketing channels")
    dup = sorted(set(channel[channel.duplicated()]))
    if dup:
        raise ValueError(f"duplicate channel row(s) for {dup}; one row per channel is required")
    values = pd.to_numeric(weights["hybrid_weight"], errors="coerce").astype(float)
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(f"missing or non-finite hybrid_weight for channel(s) {sorted(channel[bad])}")
    if (values < 0).any():
        raise ValueError(f"negative hybrid_weight for channel(s) {sorted(channel[values < 0])}")
    total = float(values.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"hybrid weights sum to {total!r}, expected 1 (tolerance {tol})")
    return pd.DataFrame({"channel": channel.to_numpy(), "hybrid_weight": values.to_numpy()})


# --------------------------------------------------------------------------- allocation
def allocate_budget(weights: pd.DataFrame, total_budget) -> pd.DataFrame:
    """Channel-level whole-rupee allocation (see ALLOCATION_COLUMNS). Amounts sum to the budget exactly."""
    budget = validate_budget(total_budget)
    w = validate_weights(weights)
    exact = [Fraction(float(x)) for x in w["hybrid_weight"]]  # exact binary value of each float
    total_weight = sum(exact, Fraction(0))
    quotas = [Fraction(budget) * f / total_weight for f in exact]
    floors = [q.numerator // q.denominator for q in quotas]
    remainders = [q - fl for q, fl in zip(quotas, floors)]
    leftover = budget - sum(floors)  # an integer in [0, number of channels)
    channels = w["channel"].tolist()
    by_largest_remainder = sorted(range(len(w)), key=lambda i: (-remainders[i], channels[i]))
    amounts = floors[:]
    for i in by_largest_remainder[:leftover]:
        amounts[i] += 1

    table = pd.DataFrame({
        "channel": channels,
        "hybrid_weight": w["hybrid_weight"].to_numpy(),
        "hybrid_percentage": 100.0 * w["hybrid_weight"].to_numpy(),
        "recommended_budget": np.array(amounts, dtype="int64"),
        "unrounded_budget": [float(q) for q in quotas],
        "rounding_adjustment": [float(a - q) for a, q in zip(amounts, quotas)],
    })
    table = table.sort_values(["recommended_budget", "channel"], ascending=[False, True]).reset_index(drop=True)
    check_allocation(table, budget)
    return table[ALLOCATION_COLUMNS]


def check_allocation(table: pd.DataFrame, total_budget: int) -> None:
    """Raise unless the allocation reconciles: integers, non-negative, adding up to the budget."""
    amounts = table["recommended_budget"]
    if (amounts < 0).any():
        raise ValueError("negative allocation produced")
    if int(amounts.sum()) != int(total_budget):
        raise ValueError(f"allocations sum to {int(amounts.sum())}, not the budget {int(total_budget)}")
    if (table["rounding_adjustment"].abs() >= 1).any():
        raise ValueError("an allocation differs from its exact share by one rupee or more")


def allocation_summary(table: pd.DataFrame, total_budget) -> dict:
    """Numbers for a 'does it reconcile?' display."""
    budget = validate_budget(total_budget)
    allocated = int(table["recommended_budget"].sum())
    return {
        "total_budget": budget,
        "allocated_total": allocated,
        "difference": budget - allocated,
        "reconciles": allocated == budget,
        "max_rounding_adjustment": float(table["rounding_adjustment"].abs().max()),
    }


# --------------------------------------------------------------------------- formatting / IO
def format_inr(amount, decimals: int = 0) -> str:
    """Indian digit grouping: 1000000 -> '₹10,00,000', 12345678 -> '₹1,23,45,678'."""
    negative = amount < 0
    text = str(abs(int(amount))) if decimals == 0 else f"{abs(float(amount)):.{decimals}f}"
    whole, _, frac = text.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if negative else "") + "₹" + whole + ("." + frac if frac else "")


def load_hybrid_attribution(path=DEFAULT_HYBRID_PATH) -> pd.DataFrame:
    """Read the Stage 11 output. Blank cells stay as text, not NaN, so only real problems trip validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Generate it with `python scripts/run_stage11.py`.")
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in ("channel", "hybrid_weight") if c not in df.columns]
    if missing:
        raise ValueError(f"hybrid attribution file is missing required column(s) {missing}")
    return df


def export_allocation(table: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 12: contribution-based budget allocation")
    parser.add_argument("--budget", default="1000000", help="total budget in rupees, e.g. 1000000 or 10,00,000")
    parser.add_argument("--hybrid", default=str(DEFAULT_HYBRID_PATH))
    parser.add_argument("--out", default=None, help="optional CSV path to save the allocation")
    args = parser.parse_args(argv)

    budget = validate_budget(parse_budget_text(args.budget))
    hybrid = load_hybrid_attribution(args.hybrid)
    table = allocate_budget(hybrid, budget)

    meta = []
    for col in ("segment", "alpha"):
        if col in hybrid.columns and hybrid[col].nunique() == 1:
            meta.append(f"{col}={hybrid[col].iloc[0]}")
    print("=== BUDGET ALLOCATION (hypothetical, contribution-based) ===")
    print(f"Hybrid weights from: {args.hybrid}  ({', '.join(meta) if meta else 'no segment/alpha info'})")
    print(f"Total budget: {format_inr(budget)}\n")
    shown = pd.DataFrame({
        "channel": table["channel"],
        "hybrid_weight": table["hybrid_weight"].round(6),
        "percentage": table["hybrid_percentage"].round(2),
        "recommended_budget": table["recommended_budget"].map(format_inr),
        "rounding_adjustment": table["rounding_adjustment"].round(3),
    })
    print(shown.to_string(index=False))
    s = allocation_summary(table, budget)
    print(f"\nAllocated total: {format_inr(s['allocated_total'])}   difference from budget: "
          f"{s['difference']}   reconciles: {s['reconciles']}")
    print("\nHow to read this:")
    for line in ALLOCATION_NOTICE:
        print(f"  - {line}")
    if args.out:
        print(f"\nSaved to {export_allocation(table, args.out)}")


if __name__ == "__main__":
    main()
