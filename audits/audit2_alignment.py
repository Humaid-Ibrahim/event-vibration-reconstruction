"""
Audit 2: Temporal alignment audit.

For each of the 30 samples in processed_data.npz, cross-correlate a global
event-rate signal (sum over the 18 channels of X) with the target y and with
the |y| envelope (event rate tracks speed magnitude, not signed velocity).

Reports per trial: lag (bins / ms) maximizing correlation, peak correlation,
and consistency of lags across trials. If a consistent nonzero lag exists,
quantifies R^2 improvement of a 1-feature linear fit (envelope ~ global rate)
after shifting by the per-trial best lag.

Saves: audits/alignment_lags.png, audits/alignment_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "processed_data.npz"

BIN_MS = 8000.0 / 2048.0          # ~3.906 ms per bin
MAX_LAG = 256                      # search +/- 1 second
SMOOTH_SIGMA_BINS = 12.8           # ~50 ms envelope smoothing

# dataviz palette (light mode)
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def xcorr_best_lag(a, b, max_lag):
    """Normalized cross-correlation of zero-mean signals a, b.
    Positive lag means a (event rate) LEADS b (vibration) by `lag` bins,
    i.e. shifting a right by `lag` maximizes the match."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L >= 0:
            x, z = a[: n - L], b[L:]
        else:
            x, z = a[-L:], b[: n + L]
        cc[i] = float(np.dot(x, z) / len(x))
    k = int(np.argmax(np.abs(cc)))
    return int(lags[k]), float(cc[k]), lags, cc


def r2_linfit(x, z):
    """R^2 of least-squares z ~ a*x + b."""
    x = x - x.mean()
    z0 = z - z.mean()
    denom = float(np.dot(x, x)) + 1e-12
    a = float(np.dot(x, z0)) / denom
    resid = z0 - a * x
    ss_tot = float(np.dot(z0, z0)) + 1e-12
    return 1.0 - float(np.dot(resid, resid)) / ss_tot


def shift(sig, lag):
    """Shift signal right by `lag` bins (truncating overlap).
    Returns the overlapping parts of (shifted_sig, companion slice indices)."""
    n = len(sig)
    if lag >= 0:
        return sig[: n - lag], slice(lag, n)
    return sig[-lag:], slice(0, n + lag)


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
    Y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
    split = ["train"] * len(data["X_train"]) + ["val"] * len(data["X_val"]) + ["test"] * len(data["X_test"])
    n_samples = X.shape[0]
    print(f"Loaded {n_samples} samples, X {X.shape}, y {Y.shape}, bin = {BIN_MS:.3f} ms")

    rows = []
    for i in range(n_samples):
        g = X[i].sum(axis=0)                     # global event-rate proxy
        yv = Y[i]
        env = gaussian_filter1d(np.abs(yv), SMOOTH_SIGMA_BINS)
        g_env = gaussian_filter1d(np.abs(g), SMOOTH_SIGMA_BINS)

        lag_y, cc_y, _, _ = xcorr_best_lag(g, yv, MAX_LAG)
        lag_e, cc_e, _, _ = xcorr_best_lag(g_env, env, MAX_LAG)

        # R^2 of linear fit envelope ~ rate, at lag 0 vs at best envelope lag
        r2_0 = r2_linfit(g_env, env)
        gs, sl = shift(g_env, lag_e)
        r2_s = r2_linfit(gs, env[sl])

        rows.append({
            "idx": i, "split": split[i],
            "lag_y_bins": lag_y, "lag_y_ms": lag_y * BIN_MS, "peak_corr_y": cc_y,
            "lag_env_bins": lag_e, "lag_env_ms": lag_e * BIN_MS, "peak_corr_env": cc_e,
            "r2_env_lag0": r2_0, "r2_env_shifted": r2_s,
        })
        print(f"[{i:2d} {split[i]:5s}] y: lag={lag_y:+4d} bins ({lag_y*BIN_MS:+7.1f} ms) corr={cc_y:+.3f} | "
              f"|y| env: lag={lag_e:+4d} ({lag_e*BIN_MS:+7.1f} ms) corr={cc_e:+.3f} | "
              f"R2(env~rate) 0-lag={r2_0:+.3f} shifted={r2_s:+.3f}")

    lags_e = np.array([r["lag_env_bins"] for r in rows])
    corrs_e = np.array([r["peak_corr_env"] for r in rows])
    lags_y = np.array([r["lag_y_bins"] for r in rows])
    corrs_y = np.array([r["peak_corr_y"] for r in rows])

    summary = {
        "bin_ms": BIN_MS,
        "envelope": {
            "median_lag_bins": float(np.median(lags_e)),
            "median_lag_ms": float(np.median(lags_e) * BIN_MS),
            "iqr_lag_bins": [float(np.percentile(lags_e, 25)), float(np.percentile(lags_e, 75))],
            "mean_abs_peak_corr": float(np.abs(corrs_e).mean()),
            "frac_abs_lag_gt_25ms": float(np.mean(np.abs(lags_e) * BIN_MS > 25)),
        },
        "signed_y": {
            "median_lag_bins": float(np.median(lags_y)),
            "mean_abs_peak_corr": float(np.abs(corrs_y).mean()),
        },
        "r2_env_lag0_mean": float(np.mean([r["r2_env_lag0"] for r in rows])),
        "r2_env_shifted_mean": float(np.mean([r["r2_env_shifted"] for r in rows])),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"|y| envelope: median lag = {summary['envelope']['median_lag_bins']:.0f} bins "
          f"({summary['envelope']['median_lag_ms']:.1f} ms), "
          f"IQR = {summary['envelope']['iqr_lag_bins']} bins, "
          f"mean |peak corr| = {summary['envelope']['mean_abs_peak_corr']:.3f}")
    print(f"signed y:     median lag = {summary['signed_y']['median_lag_bins']:.0f} bins, "
          f"mean |peak corr| = {summary['signed_y']['mean_abs_peak_corr']:.3f}")
    print(f"R2(env ~ rate): lag 0 = {summary['r2_env_lag0_mean']:.3f} -> "
          f"best-lag shifted = {summary['r2_env_shifted_mean']:.3f}")

    with open(ROOT / "audits" / "alignment_results.json", "w") as f:
        json.dump({"per_trial": rows, "summary": summary}, f, indent=2)
    print("Saved audits/alignment_results.json")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), facecolor=SURFACE)
    idx = np.arange(n_samples)
    split_bounds = [len(data["X_train"]) - 0.5, len(data["X_train"]) + len(data["X_val"]) - 0.5]

    for ax in axes:
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        for b in split_bounds:
            ax.axvline(b, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)

    axes[0].bar(idx, lags_e * BIN_MS, width=0.62, color=C_BLUE, edgecolor=SURFACE, linewidth=1)
    axes[0].axhline(0, color=INK2, linewidth=0.8)
    axes[0].set_ylabel("Best lag (ms)", color=INK2)
    axes[0].set_title("Per-trial lag maximizing corr(global event rate, |y| envelope)  —  "
                      "positive = event rate leads", color=INK, fontsize=11, loc="left")

    axes[1].bar(idx, corrs_e, width=0.62, color=C_ORANGE, edgecolor=SURFACE, linewidth=1)
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    axes[1].set_ylabel("Peak correlation", color=INK2)
    axes[1].set_xlabel("Sample index (0–17 train | 18–23 val | 24–29 test)", color=INK2)
    axes[1].set_title("Peak correlation at best lag (|y| envelope)", color=INK, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(ROOT / "audits" / "alignment_lags.png", dpi=150, facecolor=SURFACE)
    print("Saved audits/alignment_lags.png")


if __name__ == "__main__":
    main()
