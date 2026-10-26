import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "Times New Roman"

# Fig 1: architecture
fig, ax = plt.subplots(figsize=(3.4, 5.0), dpi=300)
ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")
def box(x, y, w, h, text, fc="#f2f2f2", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec="black", lw=0.7))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=5.6, weight="bold" if bold else "normal", linespacing=1.15)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", mutation_scale=7))
box(0.8, 13.6, 8.4, 1.0, "Raw event log (CSV)\nUser ID, Timestamp, Channel, Campaign, Conversion")
box(0.8, 11.9, 8.4, 1.1, "Stage 2: Validation & preprocessing\nschema, timestamps, flags, ties, stable sort")
box(0.8, 10.2, 8.4, 1.1, "Stage 3: Journey reconstruction\nconversion-delimited journeys, optional window")
box(0.0, 7.6, 4.8, 2.0, "Markov engine (Stages 4-5)\nfirst-order absorbing chain\nbaseline CR via (I - Q)h = r\nremoval effect RE_i", fc="#e6eef7")
box(5.2, 7.6, 4.8, 2.0, "Shapley engine (Stages 6-10)\n2^n coalitions; V(S) from the\nredirect-to-Null Markov game;\nexact φ_i; validation gate", fc="#eaf4ea")
box(0.8, 5.7, 8.4, 1.2, "Normalization & consistency checks\nRE_i / ΣRE;   max(φ_i, 0) / Σ max(φ, 0);   Σφ = baseline CR")
box(0.8, 4.0, 8.4, 1.1, "Stage 11: Hybrid attribution\nH_i = α · (normalized RE)_i + (1 − α) · (normalized φ)_i")
box(0.8, 2.3, 8.4, 1.1, "Stage 12: Budget allocation\nB_i = B_total w_i, largest-remainder rounding")
box(0.8, 0.5, 8.4, 1.1, "Outputs: CSV tables / console report\n(Streamlit UI: pending)", fc="#ffffff")
arrow(5, 13.6, 5, 13.0); arrow(5, 11.9, 5, 11.3)
arrow(4, 10.2, 2.4, 9.6); arrow(6, 10.2, 7.6, 9.6)
ax.annotate("", xy=(5.2, 8.6), xytext=(4.8, 8.6), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#444444", mutation_scale=5))
ax.text(5.0, 7.15, "V(S) reuses the fitted matrix P and the removal policy", fontsize=4.8, ha="center", color="#333333", style="italic", zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))
arrow(2.4, 7.6, 4, 6.9); arrow(7.6, 7.6, 6, 6.9)
arrow(5, 5.7, 5, 5.1); arrow(5, 4.0, 5, 3.4); arrow(5, 2.3, 5, 1.6)
plt.savefig("fig1_architecture.png", bbox_inches="tight", pad_inches=0.02)

# Fig 2: window sensitivity (measured, Stage 3)
labels = ["None", "24 h", "6 h"]
kept = [10000, 9444, 7539]; excl = [0, 556, 2461]
mean = [1.57, 1.48, 1.18]; single = [3926, 4109, 5319]
fig, ax1 = plt.subplots(figsize=(3.4, 2.2), dpi=300)
x = range(3)
ax1.bar(x, kept, color="#6b8fb5", width=0.55, label="Touches retained")
ax1.bar(x, excl, bottom=kept, color="#d9a46b", width=0.55, label="Touches excluded")
for i in x:
    ax1.text(i, kept[i]*0.25, f"{kept[i]:,}", ha="center", va="center", fontsize=6.5, color="white")
    if excl[i]: ax1.text(i, kept[i] + excl[i]/2, f"{excl[i]:,}", ha="center", va="center", fontsize=6.5)
ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=7); ax1.set_xlabel("Attribution window", fontsize=7)
ax1.set_ylabel("Touches", fontsize=7); ax1.tick_params(labelsize=6.5); ax1.set_ylim(0, 13500)
ax2 = ax1.twinx()
ax2.plot(list(x), mean, "k-o", ms=3, lw=0.9, label="Mean touches / journey")
for i in x: ax2.text(i + 0.1, mean[i] + 0.035, f"{mean[i]:.2f}", fontsize=6.5)
ax2.set_ylim(1.0, 1.85); ax2.set_ylabel("Mean touches per journey", fontsize=7); ax2.tick_params(labelsize=6.5)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=5.8, loc="upper center", ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
for s in ("top",): ax1.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
plt.savefig("fig2_window.png", bbox_inches="tight", pad_inches=0.02)
