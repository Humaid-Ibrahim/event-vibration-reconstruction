"""Paper figures, sized for a single IFAC column (~8.4 cm).

Figures are drawn at ~5.2 in width so 12 pt in-figure text renders at ~7.7 pt
effective in the column - at or above the 7-8 pt readability floor. Protocol
details live in the LaTeX captions, not in the images.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

BLUE, ORANGE, INK, INK2, MUTED, SURFACE = ("#2a78d6", "#eb6834", "#0b0b0b",
                                           "#52514e", "#898781", "#fcfcfb")
DPI = 220
FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT = 12, 13, 13, 11, 11

plt.rcParams.update({
    "font.size": FS_LABEL, "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
})


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=FS_TICK)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name, pad=0.08):
    fig.tight_layout()
    fig.savefig(common.ROOT / "experiments" / name, dpi=DPI,
                facecolor=SURFACE, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print("saved", name)


def inside_legend(ax):
    """Legend fully inside the axes (never clipped at the figure edge)."""
    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.96),
                    frameon=True, framealpha=0.85, edgecolor="none",
                    facecolor=SURFACE, labelcolor=INK2)
    return leg


def fig_headline():
    """Headline split only (test = trial 2): 10x10 1D CNN vs the two baselines
    (ridge refit on the same 10x10 representation)."""
    sp = json.load(open(common.ROOT / "experiments" / "headline_10x10_results.json"))
    r = sp["headline_trial2"]
    mu, sd = r["test"]["r2"]
    rid = r["baselines"]["ridge10x10"]["test"]["r2"]
    mean_b = r["baselines"]["per_timestep_mean"]["test"]["r2"]

    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURFACE)
    style(ax)
    ax.bar([0], [mu], 0.6, yerr=[sd], color=BLUE, edgecolor=SURFACE, linewidth=1,
           error_kw={"ecolor": INK2, "capsize": 4, "elinewidth": 1.4})
    ax.bar([1, 2], [rid, mean_b], 0.6, color=ORANGE, edgecolor=SURFACE, linewidth=1)
    for x, v in ((0, mu), (1, rid), (2, mean_b)):
        ax.annotate(f"{v:.2f}", xy=(x, v), xytext=(x + (0.24 if x == 0 else 0),
                    v + 0.025), ha="center", fontsize=FS_ANNOT, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1D CNN", "ridge", "mean\nsignal"], fontsize=FS_TICK)
    ax.set_ylabel("Test R²")
    ax.set_ylim(0, 0.95)
    save(fig, "fig_headline_r2.png")


def fig_cv():
    """Per-fold CV bars (10x10 model vs ridge-on-10x10) + pooled-mean markers."""
    sp = json.load(open(common.ROOT / "experiments" / "headline_10x10_results.json"))
    pf = sp["cv"]["per_fold"]
    folds = [f"fold{k}" for k in range(1, 6)]
    mu = [pf[f]["r2"][0] for f in folds]
    sd = [pf[f]["r2"][1] for f in folds]
    rid = sp["cv"]["ridge10x10"]["per_fold_r2"]
    pooled_m = sp["cv"]["test_overall"]["r2"][0]
    pooled_r = sp["cv"]["ridge10x10"]["test_overall"]["r2"][0]

    fig, ax = plt.subplots(figsize=(5.4, 3.5), facecolor=SURFACE)
    style(ax)
    xs = np.arange(5)
    ax.bar(xs - 0.19, mu, 0.36, yerr=sd, color=BLUE, edgecolor=SURFACE,
           error_kw={"ecolor": INK2, "capsize": 3, "elinewidth": 1.2},
           label="1D CNN")
    ax.bar(xs + 0.19, rid, 0.36, color=ORANGE, edgecolor=SURFACE, label="ridge")
    ax.axhline(pooled_m, color=BLUE, linewidth=1.3, linestyle="--", alpha=0.85,
               label=f"pooled {pooled_m:.2f}")
    ax.axhline(pooled_r, color=ORANGE, linewidth=1.3, linestyle="--", alpha=0.85,
               label=f"pooled {pooled_r:.2f}")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"trial {k}" for k in range(1, 6)], fontsize=FS_TICK)
    ax.set_xlabel("Held-out test trial (fold)")
    ax.set_ylabel("Test R²")
    ax.set_ylim(0, 1.0)
    # legend fully OUTSIDE the axes (above) - the plot area is too crowded
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=4, labelcolor=INK2, columnspacing=1.0, handlelength=1.4,
              handletextpad=0.5)
    save(fig, "fig_cv_r2.png")


def fig_ablation():
    """ROI-grid sweep + polarity ablation (headline split, 3 slots each)."""
    ga = json.load(open(common.ROOT / "experiments" / "grid_ablation_results.json"))
    g = ga["grids"]
    order = ["grid1x1", "grid3x3_nopol", "grid3x3_pol", "grid5x5", "grid10x10"]
    labels = ["1×1", "3×3\nno pol.", "3×3", "5×5", "10×10"]
    mu = [g[m]["test"]["r2"][0] for m in order]
    sd = [g[m]["test"]["r2"][1] for m in order]
    # polarity-split bars in blue; the no-polarity variant in a light tint
    TINT = "#86b6ef"
    colors = [BLUE, TINT, BLUE, BLUE, BLUE]

    fig, ax = plt.subplots(figsize=(5.3, 3.5), facecolor=SURFACE)
    style(ax)
    xs = np.arange(len(order))
    ax.bar(xs, mu, 0.62, yerr=sd, color=colors, edgecolor=SURFACE, linewidth=1,
           error_kw={"ecolor": INK2, "capsize": 4, "elinewidth": 1.4})
    for x, v in zip(xs, mu):
        ax.annotate(f"{v:.2f}", xy=(x, max(v, 0) + 0.035), ha="center",
                    fontsize=FS_ANNOT, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("ROI grid (polarity-split unless noted)")
    ax.set_ylabel("Test R²")
    ax.set_ylim(0, 0.95)
    save(fig, "fig_ablation.png")


def fig_example():
    """Example reconstruction: time domain + spectrum, stacked."""
    d = np.load(common.ROOT / "experiments" / "predictions_simple1dcnn_10x10.npz")
    yt, yp, fs = d["y_test"], d["y_test_pred"], float(d["fs"])
    from sklearn.metrics import r2_score
    r2s = [r2_score(yt[i], yp[i]) for i in range(len(yt))]
    i = int(np.argsort(r2s)[len(r2s) // 2])
    t = np.arange(yt.shape[1]) / fs

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.6), facecolor=SURFACE)
    for ax in axes:
        style(ax)
    axes[0].plot(t, yt[i], color=INK2, linewidth=1.0, label="LDV")
    axes[0].plot(t, yp[i], color=BLUE, linewidth=1.0, alpha=0.9, label="1D CNN")
    axes[0].set_xlim(0, 8)
    yl = float(np.abs(np.concatenate([yt[i], yp[i]])).max())
    axes[0].set_ylim(-1.15 * yl, 1.35 * yl)  # headroom for the legend
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Velocity")
    inside_legend(axes[0])

    freqs = np.fft.rfftfreq(yt.shape[1], 1 / fs)
    band = freqs <= 40
    axes[1].plot(freqs[band], np.abs(np.fft.rfft(yt[i]))[band], color=INK2,
                 linewidth=1.2, label="LDV")
    axes[1].plot(freqs[band], np.abs(np.fft.rfft(yp[i]))[band], color=BLUE,
                 linewidth=1.2, alpha=0.9, label="1D CNN")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("|FFT|")
    smax = float(np.abs(np.fft.rfft(yt[i]))[band].max())
    axes[1].set_ylim(0, 1.22 * smax)  # headroom for the legend
    inside_legend(axes[1])
    save(fig, "fig_example_pred.png")


if __name__ == "__main__":
    fig_headline()
    fig_cv()
    fig_ablation()
    fig_example()
