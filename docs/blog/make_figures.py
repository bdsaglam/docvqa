"""Generate the data figures for the DocVQA active-perception blog post.

Numbers are the current-code validation results (Qwen 3.5 27B homog, val 25-doc/
80-Q, n=8), verified against docs/results.md / docs/pass-at-k.md. Descriptive
labels only (no engineering solver names), matching the post's prose.

Run: uv run python workspace/docvqa-repl-blog-2026-06-30/figures/make_figures.py
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

# Muted, print-friendly palette
C_TOP = "#2f6f9f"      # REPL + active-perception call
C_MID = "#d9883b"      # missing one half
C_FLOOR = "#9aa0a6"    # OCR-only / no-scaffold anchors
GRID = "#d7d7d7"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- F3: tiers
def fig3_tiers():
    # (label, mean, std, tier)
    rows = [
        ("RLM (full method)", 41.88, 5.79, "top"),
        ("CodeAct (append-only twin)", 39.53, 2.83, "top"),
        ("+ general sub-agent", 36.72, 2.75, "top"),
        ("+ OCR & search", 36.56, 2.89, "top"),
        ("ReAct (no REPL)", 27.19, 3.19, "mid"),
        ("In-context pixels (no sub-VLM)", 22.34, 2.79, "mid"),
        ("Raw multi-image (no scaffold)", 20.94, 1.60, "mid"),
        ("Competition prompt (no scaffold)", 18.91, 1.94, "floor"),
        ("OCR-only (no vision)", 14.69, 2.19, "floor"),
    ]
    color = {"top": C_TOP, "mid": C_MID, "floor": C_FLOOR}
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    cols = [color[r[3]] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = range(len(rows))
    ax.barh(y, means, xerr=stds, color=cols, height=0.68,
            error_kw=dict(ecolor="#555555", capsize=3, lw=1))
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("DocVQA-2026 val accuracy (ANLS %), mean ± std over 8 trials")
    ax.set_xlim(0, 50)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.6, i, f"{m:.1f}", va="center", fontsize=9.5,
                color="#333333")
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    legend = [
        Patch(color=C_TOP, label="REPL + active-perception call"),
        Patch(color=C_MID, label="missing one half"),
        Patch(color=C_FLOOR, label="no-scaffold / no-vision floor"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Three clean tiers", x=0.04, ha="left", fontsize=13,
                 fontweight="bold")
    save(fig, "f3-tiers.png")


# ---------------------------------------------------------------- F4: 2x2
def fig4_grid():
    # rows: with / without REPL ; cols: with / without active-perception call
    vals = [[41.9, 22.3],   # with REPL
            [27.2, 20.9]]   # without REPL
    notes = [["full method", "display() only"],
             ["ReAct", "raw multi-image"]]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    # color by value
    for r in range(2):
        for c in range(2):
            v = vals[r][c]
            top = v > 35
            face = C_TOP if top else C_MID if v > 24 else C_FLOOR
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=face,
                                       edgecolor="white", lw=3, alpha=0.92))
            ax.text(c + 0.5, 1 - r + 0.60, f"{v:.0f}%", ha="center",
                    va="center", fontsize=22, fontweight="bold", color="white")
            ax.text(c + 0.5, 1 - r + 0.30, notes[r][c], ha="center",
                    va="center", fontsize=10, color="white")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["active-perception call", "no call\n(pixels in-context / none)"])
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["REPL", "no REPL"], rotation=90, va="center")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Both halves are needed; neither alone suffices",
                 fontsize=12, fontweight="bold", pad=10)
    save(fig, "f4-2x2.png")


# ----------------------------------------------------- F5: VLM-swap (perception)
def fig5_matrix():
    # Reasoner x Perceiver 3x3 (rvlm, val ANLS, mean over n trials).
    # Source: docs/experiments/rvlm-reasoner-perceiver-3x3.md. Two cells not run.
    import numpy as np
    sizes = ["4B", "9B", "27B"]
    # rows = reasoner (bottom-to-top 4B..27B after flip), cols = perceiver VLM
    mean = np.array([[14.22, np.nan, 21.09],
                     [np.nan, 18.91, 25.31],
                     [32.81, 37.2, 41.88]])
    std = np.array([[3.83, np.nan, 3.16],
                    [np.nan, 3.81, 4.16],
                    [3.13, 6.2, 5.79]])
    n = np.array([[8, 0, 8], [0, 8, 8], [4, 4, 8]])

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    masked = np.ma.masked_invalid(mean)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#efefef")
    im = ax.imshow(masked, cmap=cmap, vmin=8, vmax=48, origin="lower")
    for i in range(3):
        for j in range(3):
            if np.isnan(mean[i, j]):
                ax.text(j, i, "not run", ha="center", va="center",
                        fontsize=10, color="#999999", style="italic")
            else:
                dark = mean[i, j] > 30
                ax.text(j, i, f"{mean[i, j]:.1f}",
                        ha="center", va="center", fontsize=15,
                        fontweight="bold",
                        color="white" if dark else "#1f3d57")
                ax.text(j, i - 0.30, f"±{std[i, j]:.1f}  (n={n[i, j]})",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if dark else "#4a6a88")
    ax.set_xticks(range(3), sizes)
    ax.set_yticks(range(3), sizes)
    ax.set_xlabel("Perceiver (VLM behind the perception call)")
    ax.set_ylabel("Reasoner (drives the REPL)")
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Reasoner × VLM: validation accuracy (ANLS %)",
                 fontsize=12, fontweight="bold")
    save(fig, "f5-matrix.png")


# ------------------------------------------------------- F-cat: per-category gap
def figcat():
    # Per-category rvlm - react gap, Qwen 3.5 27B homog, 8 trials (binary acc).
    rows = [
        ("engineering drawing", 36.2),
        ("business report", 30.0),
        ("infographic", 18.8),
        ("comics", 10.0),
        ("science poster", 10.0),
        ("maps", 7.5),
        ("science paper", 3.7),
        ("slide", 1.2),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = range(len(rows))
    ax.barh(y, vals, color=C_TOP, height=0.66)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"+{v:.0f}", va="center", fontsize=9.5,
                color="#333333")
    ax.set_xlabel("active-perception advantage over ReAct (pp)")
    ax.set_xlim(0, 40)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("The advantage tracks visual density", fontsize=12,
                 fontweight="bold")
    save(fig, "f-category.png")


# ---------------------------------------------------------------- F1: composite
def fig1_composite():
    full = plt.imread(os.path.join(HERE, "f1-nvidia-full.png"))
    crop = plt.imread(os.path.join(HERE, "f1-nvidia-crop.png"))
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.0, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    for ax, img, title in [(axl, full, "full page → wrong number"),
                           (axr, crop, "crop & zoom → right number")]:
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.92, bottom=0.01)
    save(fig, "f1-nvidia-chart.png")


# ------------------------------------------------------------- F2: architecture
def fig2_architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 5.2))

    def box(ax, xy, w, h, text, fc):
        ax.add_patch(FancyBboxPatch(
            (xy[0] - w / 2, xy[1] - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor="#33475b", facecolor=fc))
        ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=10.5,
                color="#1b2a38")

    def arrow(ax, p0, p1, label="", rad=0.0, side="right", style="-|>"):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=14, lw=1.4,
            color="#33475b", connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            dx = 0.30 if side == "right" else -0.30
            ax.text(mx + dx, my, label, fontsize=9, ha="left" if side == "right"
                    else "right", color="#2f6f9f", style="italic")

    for ax in (axl, axr):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 6)
        ax.axis("off")

    # Left: Perceive-Reason-Code
    axl.set_title("Perceive-Reason-Code", fontsize=12, fontweight="bold")
    box(axl, (2, 5.2), 2.6, 0.8, "Reasoner (LLM)", "#dbe7f1")
    box(axl, (2, 3.5), 2.6, 0.8, "Python REPL", "#eef3f8")
    box(axl, (2, 1.8), 2.6, 0.8, "Perceiver (VLM)", "#dbe7f1")
    box(axl, (2, 0.4), 2.8, 0.55, "document pages", "#f1f1f1")
    arrow(axl, (1.7, 4.8), (1.7, 3.9), "writes code", side="left")
    arrow(axl, (2.3, 3.9), (2.3, 4.8), "reads variables", side="right")
    arrow(axl, (1.7, 3.1), (1.7, 2.2), "look(crop, zoom)", side="left")
    arrow(axl, (2.0, 1.4), (2.0, 0.7))
    arrow(axl, (2.3, 2.2), (2.3, 3.1), "observation (text)", rad=-0.0, side="right")

    # Right: ReAct
    axr.set_title("ReAct", fontsize=12, fontweight="bold")
    box(axr, (2, 5.2), 2.6, 0.8, "Reasoner (LLM)", "#dbe7f1")
    box(axr, (2, 2.6), 2.6, 0.8, "Perceiver (VLM)", "#dbe7f1")
    box(axr, (2, 0.9), 2.8, 0.55, "document pages", "#f1f1f1")
    arrow(axr, (1.7, 4.8), (1.7, 3.0), "tool call", side="left")
    arrow(axr, (2.3, 3.0), (2.3, 4.8), "observation (text)", side="right")
    arrow(axr, (2.0, 2.2), (2.0, 1.2), "whole page", side="right")
    axr.text(2, 0.1, "no crop or zoom; perception fixed at page granularity",
             ha="center", fontsize=8.5, color="#8a5a2b", style="italic")

    fig.subplots_adjust(wspace=0.12, top=0.9)
    save(fig, "f2-architecture.png")


def fig_lengthaxis():
    """Blog Figure 4: recursive-perception advantage vs raw multi-image baseline
    across two benchmarks of different length. Baseline left, ours (blue) right."""
    import numpy as np
    groups = ["MP-DocVQA\n(short, ≤20 pg)", "MMLongBench-Doc\n(long, ~47 pg)"]
    ours, ours_s = [61.8, 66.6], [1.79, 2.15]     # active perception (rvlm)
    base, base_s = [58.1, 24.2], [0.81, 0.60]     # raw multi-image, no scaffold
    gaps = [4, 42]
    C_BASE = "#c7bca6"   # muted tan
    RED = "#9a3b2e"

    x = np.arange(len(groups))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.bar(x - w / 2, base, w, yerr=base_s, label="raw multi-image baseline (no scaffold)",
           color=C_BASE, error_kw=dict(ecolor="#444", capsize=3, lw=1))
    ax.bar(x + w / 2, ours, w, yerr=ours_s, label="RLM (ours)",
           color=C_TOP, error_kw=dict(ecolor="#444", capsize=3, lw=1))
    for i in range(len(groups)):
        ax.text(x[i] - w / 2, base[i] + base_s[i] + 1.4, f"{base[i]:.0f}",
                ha="center", fontsize=11, color="#444")
        ax.text(x[i] + w / 2, ours[i] + ours_s[i] + 1.4, f"{ours[i]:.0f}",
                ha="center", fontsize=11, color="#444")
        # gap label above the group (text only, consistent across both groups)
        ax.text(x[i], max(ours[i], base[i]) + max(ours_s[i], base_s[i]) + 5.5,
                f"+{gaps[i]} pp", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 92)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("Across benchmarks, the advantage widens on longer documents",
                 fontsize=13, fontweight="bold", pad=30)
    fig.text(0.5, -0.01,
             'raw baseline “Unknown” rate climbs 22% → 87% as evidence '
             'falls off the fixed page budget',
             ha="center", fontsize=10, style="italic", color="#8a6a2b")
    save(fig, "f-lengthaxis.png")


if __name__ == "__main__":
    fig1_composite()
    fig2_architecture()
    fig3_tiers()
    fig4_grid()
    fig5_matrix()
    figcat()
    fig_lengthaxis()
    print("done")
