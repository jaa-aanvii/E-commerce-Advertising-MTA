"""Heatmap figures for Section VII-E/F, drawn from the CSVs written by paper_results.py."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

def heatmap(matrix, shares, rows, cols, xlabel, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(3.4, 0.36 * len(rows) + 1.1), dpi=300)
    im = ax.imshow(matrix, cmap="Blues_r", vmin=1, vmax=6, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            r = int(matrix[i, j])
            txt = f"{r}" if shares is None else f"{r}\n{shares[i, j]:.2f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6 if shares is not None else 6.5,
                    color="white" if r <= 2 else "black", linespacing=1.0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=6.5)
    ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, ticks=range(1, 7))
    cb.ax.invert_yaxis(); cb.ax.tick_params(labelsize=6); cb.set_label("Rank (1 = highest share)", fontsize=6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)

# alpha sweep: rows = alpha, cols = channel
rows = list(csv.DictReader(open("fig3_alpha_sweep.csv")))
chs = sorted({r["channel"] for r in rows}); alphas = sorted({float(r["alpha"]) for r in rows})
R = np.array([[int(next(r["rank"] for r in rows if float(r["alpha"]) == a and r["channel"] == c)) for c in chs] for a in alphas])
heatmap(R, None, [f"{a:.1f}" for a in alphas], chs, "Channel", "α (weight on Markov)",
        "Hybrid rank of each channel across α", "alpha_rank_heatmap.png")

# windows: rows = window, cols = channel, rank + hybrid share
rows = list(csv.DictReader(open("window_sensitivity.csv")))
wins = ["none", "24h", "6h"]
R = np.array([[int(next(r["h_rank"] for r in rows if r["window"] == w and r["channel"] == c)) for c in chs] for w in wins])
S = np.array([[float(next(r["h_pct"] for r in rows if r["window"] == w and r["channel"] == c)) for c in chs] for w in wins])
heatmap(R, S, ["No window", "24 h", "6 h"], chs, "Channel", "Attribution window",
        "Hybrid rank and share (α = 0.5) by window", "window_rank_heatmap.png")
