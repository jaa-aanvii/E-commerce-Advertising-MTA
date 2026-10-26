"""Generate the numbers for Section VII of the paper that the stage scripts do not print.

Everything is computed in memory from data/raw/multi_touch_attribution_data.csv with the project's
own modules, so it uses exactly the same definitions as Stages 2-12. Nothing in data/processed is
read or overwritten. Outputs go to paper/results/.

Run from the project root:   .venv\\Scripts\\python.exe paper\\paper_results.py

Produces (one CSV per paper table, plus a console summary):
  table6_markov.csv        Table VI    CR without, RE, M-hat, rank
  table7_shapley.csv       Table VII   phi, phi-hat, rank (+ efficiency gap in the summary)
  table8_hybrid.csv        Table VIII  M-hat, phi-hat, H at alpha = 0.5, rank (+ rho, tau)
  fig3_alpha_sweep.csv     Fig. 3      H for alpha = 0.0 .. 1.0 (PNG too if matplotlib is installed)
  window_sensitivity.csv   Sec. VII-F  shares and ranks under none / 24 h / 6 h (+ rho, tau vs none)
  table9_baselines.csv     Table IX    first / last / linear / Markov / Shapley / hybrid shares
  table10_budget.csv       Table X     allocation of Rs 10,00,000 at alpha = 0.5
  table11_runtime.csv      Table XI    median wall-clock times over repeated runs
"""
from __future__ import annotations

import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from budget_allocator import allocate_budget  # noqa: E402
from hybrid_engine import hybrid_attribution  # noqa: E402
from mta.data_pipeline import KAGGLE_CONFIG, clean_events  # noqa: E402
from mta.journey_builder import PATH_SEP, build_journeys  # noqa: E402
from mta.markov_engine import fit_markov_model, fit_segment_models, model_removal_effects  # noqa: E402
from shapley_engine.coalition_generator import (  # noqa: E402
    coalitions_to_frame, extract_channels_from_journeys, generate_coalitions)
from shapley_engine.coalition_value import calculate_all_coalition_values  # noqa: E402
from shapley_engine.shapley_engine import build_value_lookup, compute_shapley_values  # noqa: E402

OUT = ROOT / "paper" / "results"
RAW = ROOT / "data" / "raw" / "multi_touch_attribution_data.csv"
ALPHAS = [round(0.1 * k, 1) for k in range(11)]
WINDOWS = [None, 24, 6]
BUDGET = 10_00_000
TIMING_REPEATS = 5


# --------------------------------------------------------------------------- ranking statistics
def ranks(scores: pd.Series) -> pd.Series:
    """Rank 1 = largest score; equal scores (to 12 decimals) ordered by channel name, as in the pipeline."""
    order = sorted(scores.index, key=lambda c: (-round(float(scores[c]), 12), c))
    return pd.Series({c: i for i, c in enumerate(order, start=1)})


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman's rho on average ranks (pandas handles ties)."""
    return float(a.rank().corr(b.reindex(a.index).rank()))


def kendall_tau_b(a: pd.Series, b: pd.Series) -> float:
    """Kendall's tau-b, written out because scipy is not a project dependency."""
    b = b.reindex(a.index)
    keys = list(a.index)
    conc = disc = ties_a = ties_b = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            da = a[keys[i]] - a[keys[j]]
            db = b[keys[i]] - b[keys[j]]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    denom = ((conc + disc + ties_a) * (conc + disc + ties_b)) ** 0.5
    return (conc - disc) / denom if denom else float("nan")


# --------------------------------------------------------------------------- one full attribution run
def attribute(journeys: pd.DataFrame, alpha: float = 0.5) -> dict:
    """Markov RE, Shapley phi and the hybrid table for one set of journeys (same code as Stages 4-11)."""
    model = fit_markov_model(journeys)
    markov = model_removal_effects(model)
    channels = extract_channels_from_journeys(journeys)
    values = calculate_all_coalition_values(model.probs, coalitions_to_frame(generate_coalitions(channels)))
    lookup = build_value_lookup(values)
    phi = compute_shapley_values(lookup)
    shapley = pd.DataFrame({"channel": list(phi), "shapley_value": list(phi.values())})
    hybrid = hybrid_attribution(markov, shapley, alpha=alpha)
    return {"model": model, "markov": markov, "phi": phi, "lookup": lookup, "shapley": shapley, "hybrid": hybrid}


# --------------------------------------------------------------------------- heuristic baselines
def heuristic_shares(journeys: pd.DataFrame) -> pd.DataFrame:
    """First-, last-touch and linear credit on CONVERTING journeys only, normalised to shares.

    Linear gives 1/k to each of the k touches, so a channel appearing twice in a journey gets 2/k.
    """
    credit = {"first_touch": defaultdict(float), "last_touch": defaultdict(float), "linear": defaultdict(float)}
    for path in journeys.loc[journeys["converted"] == 1, "path"]:
        touches = path.split(PATH_SEP)[1:-1]  # drop Start and the terminal state
        credit["first_touch"][touches[0]] += 1.0
        credit["last_touch"][touches[-1]] += 1.0
        for ch in touches:
            credit["linear"][ch] += 1.0 / len(touches)
    out = pd.DataFrame(credit).fillna(0.0)
    return out / out.sum()


def timed(fn, repeats: int = TIMING_REPEATS) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    events = clean_events(RAW, KAGGLE_CONFIG).clean
    journeys = build_journeys(events, window_hours=None).journeys
    base = attribute(journeys, alpha=0.5)
    h = base["hybrid"].set_index("channel")
    channels = sorted(h.index)

    # ---- Table VI: Markov
    m = base["markov"].set_index("channel")
    t6 = pd.DataFrame({
        "cr_without": m["conversion_probability_without_channel"],
        "removal_effect": m["removal_effect"],
        "m_hat_pct": 100 * h["markov_weight"],
    }).loc[channels]
    t6["rank"] = ranks(t6["removal_effect"])
    t6.to_csv(OUT / "table6_markov.csv")
    segs = fit_segment_models(journeys)

    # ---- Table VII: Shapley
    phi = pd.Series(base["phi"])
    lookup = base["lookup"]
    v_n, v_empty = lookup[frozenset(channels)], lookup[frozenset()]
    t7 = pd.DataFrame({"phi_raw": phi, "phi_hat_pct": 100 * h["shapley_weight"]}).loc[channels]
    t7["rank"] = ranks(t7["phi_raw"])
    t7.to_csv(OUT / "table7_shapley.csv")

    # ---- Table VIII: hybrid at 0.5 + rank agreement between the two engines
    t8 = pd.DataFrame({"m_hat_pct": 100 * h["markov_weight"], "phi_hat_pct": 100 * h["shapley_weight"],
                       "h_pct": h["hybrid_percentage"]}).loc[channels]
    t8["rank"] = ranks(t8["h_pct"])
    t8.to_csv(OUT / "table8_hybrid.csv")
    rho_mp = spearman(t6["removal_effect"], t7["phi_raw"])
    tau_mp = kendall_tau_b(t6["removal_effect"], t7["phi_raw"])

    # ---- Fig. 3: alpha sweep (re-weights the same two vectors, exactly as hybrid_engine does)
    sweep = []
    for a in ALPHAS:
        ha = hybrid_attribution(base["markov"], base["shapley"], alpha=a).set_index("channel")
        r = ranks(ha["hybrid_weight"])
        for c in channels:
            sweep.append({"alpha": a, "channel": c, "hybrid_pct": ha.loc[c, "hybrid_percentage"], "rank": r[c]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "fig3_alpha_sweep.csv", index=False)
    rank_table = sweep.pivot(index="alpha", columns="channel", values="rank")
    swaps = [f"{prev:.1f}->{cur:.1f}" for prev, cur in zip(ALPHAS, ALPHAS[1:])
             if not rank_table.loc[prev].equals(rank_table.loc[cur])]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
        for c in channels:
            s = sweep[sweep["channel"] == c]
            ax.plot(s["alpha"], s["hybrid_pct"], marker="o", ms=2.5, lw=0.9, label=c)
        ax.set_xlabel("α (weight on Markov)", fontsize=7)
        ax.set_ylabel("Hybrid share (%)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=5.5, frameon=False, ncol=2)
        fig.savefig(OUT / "fig3_alpha_sweep.png", bbox_inches="tight", pad_inches=0.02)
        fig_note = "fig3_alpha_sweep.png written"
    except ImportError:
        fig_note = "matplotlib not installed: plot fig3_alpha_sweep.csv yourself, or pip install matplotlib"

    # ---- Section VII-F: windows (journeys rebuilt per window, then Stages 4-11 rerun)
    win_rows, win_stats = [], []
    ref = None
    for w in WINDOWS:
        res = attribute(build_journeys(events, window_hours=w).journeys, alpha=0.5)
        hw = res["hybrid"].set_index("channel").loc[channels]
        label = "none" if w is None else f"{w}h"
        r = ranks(hw["hybrid_weight"])
        for c in channels:
            win_rows.append({"window": label, "channel": c, "m_hat_pct": 100 * hw.loc[c, "markov_weight"],
                             "phi_hat_pct": 100 * hw.loc[c, "shapley_weight"],
                             "h_pct": hw.loc[c, "hybrid_percentage"], "h_rank": r[c]})
        if ref is None:
            ref = hw["hybrid_weight"]
        win_stats.append({"window": label, "baseline_cr": res["model"].base_conversion_probability,
                          "rho_vs_none": spearman(ref, hw["hybrid_weight"]),
                          "tau_vs_none": kendall_tau_b(ref, hw["hybrid_weight"])})
    pd.DataFrame(win_rows).to_csv(OUT / "window_sensitivity.csv", index=False)
    win_stats = pd.DataFrame(win_stats)
    win_stats.to_csv(OUT / "window_rank_agreement.csv", index=False)

    # ---- Table IX: heuristic baselines next to the three models
    t9 = (100 * heuristic_shares(journeys)).reindex(channels).fillna(0.0)
    t9["markov"], t9["shapley"], t9["hybrid"] = t8["m_hat_pct"], t8["phi_hat_pct"], t8["h_pct"]
    t9.to_csv(OUT / "table9_baselines.csv")
    t9_ranks = t9.apply(ranks)
    agree = pd.DataFrame({m_: {"rho_vs_hybrid": spearman(t9["hybrid"], t9[m_]),
                               "tau_vs_hybrid": kendall_tau_b(t9["hybrid"], t9[m_])} for m_ in t9.columns}).T
    agree.to_csv(OUT / "table9_rank_agreement.csv")

    # ---- Table X: budget
    t10 = allocate_budget(base["hybrid"], BUDGET)
    t10.to_csv(OUT / "table10_budget.csv", index=False)

    # ---- Table XI: runtime (median of repeated runs)
    model = fit_markov_model(journeys)
    t_markov = timed(lambda: model_removal_effects(fit_markov_model(journeys)))
    t_shapley = timed(lambda: compute_shapley_values(build_value_lookup(calculate_all_coalition_values(
        model.probs, coalitions_to_frame(generate_coalitions(extract_channels_from_journeys(journeys)))))))
    t11 = pd.DataFrame([
        {"quantity": "Markov fit + removal effects (s)", "value": t_markov},
        {"quantity": "Coalition values + Shapley (s)", "value": t_shapley},
        {"quantity": "repeats (median reported)", "value": TIMING_REPEATS},
        {"quantity": "python", "value": platform.python_version()},
        {"quantity": "processor (as reported by OS)", "value": platform.processor()},
        {"quantity": "os", "value": platform.platform()},
    ])
    t11.to_csv(OUT / "table11_runtime.csv", index=False)

    # ---- console summary
    fmt = lambda x: f"{x:.6f}"  # noqa: E731
    print("=== TABLE VI: Markov (all journeys, no window) ===")
    print(t6.to_string(float_format=fmt))
    print(f"Sum of removal effects: {t6['removal_effect'].sum():.6f}")
    print(f"Sparse states (all): {segs.models['all'].sparse_states or 'none'}")
    for name, mod in segs.models.items():
        print(f"  segment {name}: journeys {mod.n_journeys}, baseline CR {mod.base_conversion_probability:.6f}, "
              f"sparse {mod.sparse_states or 'none'}")
    for name, why in segs.skipped.items():
        print(f"  segment {name} NOT fitted: {why}")
    print("\n=== TABLE VII: Shapley ===")
    print(t7.to_string(float_format=fmt))
    print(f"sum(phi) = {phi.sum():.12f}, V(N) = {v_n:.12f}, V(empty) = {v_empty:.12f}, "
          f"efficiency gap = {phi.sum() - (v_n - v_empty):.3e}")
    print(f"Clipped (negative) Shapley values: {int(h['shapley_clipped'].sum())}")
    print("\n=== TABLE VIII: hybrid, alpha = 0.5 ===")
    print(t8.to_string(float_format=fmt))
    print(f"Markov vs Shapley ranking: Spearman rho = {rho_mp:.4f}, Kendall tau-b = {tau_mp:.4f}")
    print("\n=== FIG. 3: alpha sweep, ranks per alpha ===")
    print(rank_table.to_string())
    print(f"Rank changes between consecutive alphas: {', '.join(swaps) if swaps else 'none'}")
    print(fig_note)
    print("\n=== SECTION VII-F: windows ===")
    print(pd.DataFrame(win_rows).pivot(index="channel", columns="window", values="h_pct")
          [["none", "24h", "6h"]].to_string(float_format=fmt))
    print(win_stats.to_string(index=False, float_format=fmt))
    print("\n=== TABLE IX: shares (%) by method ===")
    print(t9.to_string(float_format=lambda x: f"{x:.2f}"))
    print("Ranks:")
    print(t9_ranks.to_string())
    print(agree.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== TABLE X: budget ===")
    print(t10.to_string(index=False))
    print(f"Allocated total: {int(t10['recommended_budget'].sum())} (budget {BUDGET})")
    print("\n=== TABLE XI: runtime ===")
    print(t11.to_string(index=False))
    print(f"\nCSV files written to {OUT}")


if __name__ == "__main__":
    main()
