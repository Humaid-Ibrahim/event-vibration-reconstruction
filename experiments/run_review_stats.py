"""
Review-fix statistics for the SAFEPROCESS revision.

 1. Paired head-to-head sign/Wilcoxon tests, simple1dcnn vs REF-EvSet
    (6 test recordings, experiments/refevset_comparison.json)
 2. 95% bootstrap CIs (10k resamples): fixed-split 8-seed mean test R2,
    stable-band (non-collapsed) mean, pooled CV R2 (simple1dcnn 40 runs,
    ridge 5 folds)
 3. Event-sparsity verification at several granularities (processed X,
    REF-EvSet level-1 caches: 16x16px patches @1ms, our 3x3 ROI grid @4ms)
 4. Dominant frequency of all 30 recordings (On_set3_trail1 anomaly check)

Output: experiments/review_stats.json, printed summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

rng = np.random.default_rng(0)
N_BOOT = 10_000
out = {}


# ------------------------------------------------------------------ 1. paired
def sign_test_p(n_pos, n):
    """Two-sided exact binomial sign test (p=0.5), ignoring ties upstream."""
    k = min(n_pos, n - n_pos)
    p = 2 * sum(stats.binom.pmf(i, n, 0.5) for i in range(0, k + 1))
    return float(min(p, 1.0))


j = json.load(open(HERE / "refevset_comparison.json"))
ours = {r["name"]: r for r in j["per_recording"]["ours"]}
ref = {r["name"].replace("trai4", "trail4"): r for r in j["per_recording"]["refevset"]}
names = sorted(ours)
HIGHER = {"r2": True, "spec_nrmse": False, "decay_err": False, "env_r2": True}
paired = {}
for k, hi in HIGHER.items():
    d = np.array([ours[n][k] - ref[n][k] for n in names])
    wins = int(((d > 0) if hi else (d < 0)).sum())
    n = len(d)
    try:
        w_p = float(stats.wilcoxon(d, alternative="two-sided", mode="exact").pvalue)
    except Exception:
        w_p = None
    paired[k] = {"diffs_ours_minus_ref": [round(float(x), 4) for x in d],
                 "ours_wins": wins, "n": n,
                 "sign_test_p": sign_test_p(wins, n), "wilcoxon_p": w_p,
                 "direction_better": "higher" if hi else "lower"}
out["paired_headtohead_test6"] = {"recordings": names, "metrics": paired}

# --------------------------------------------------------------- 2. bootstrap
def boot_ci(x, stat=np.mean):
    x = np.asarray(x, dtype=float)
    bs = stat(rng.choice(x, size=(N_BOOT, len(x))), axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


s = json.load(open(HERE / "stability_results.json"))
r2_seeds = [m["r2"] for m in s["test_per_seed"]]
stable = [r for r in r2_seeds if r >= 0.1]
c = json.load(open(HERE / "cv_warmup_results.json"))
cv_runs = c["simple1dcnn"]["pooled_clean"]["runs"]
ridge_folds = c["ridge"]["per_fold_r2_clean"]
out["bootstrap_ci_95"] = {
    "n_resamples": N_BOOT,
    "fixed_split_8seed_mean_r2": {"mean": float(np.mean(r2_seeds)), "ci": boot_ci(r2_seeds), "n": len(r2_seeds)},
    "fixed_split_stable_band_mean_r2": {"mean": float(np.mean(stable)), "ci": boot_ci(stable), "n": len(stable),
                                        "definition": "seeds with test R2 >= 0.1"},
    "cv_pooled_simple1dcnn_r2": {"mean": float(np.mean(cv_runs)), "ci": boot_ci(cv_runs), "n": len(cv_runs),
                                 "definition": "40 fold x seed runs, fold 3 scored excl. On_set2_trail3"},
    "cv_pooled_ridge_r2": {"mean": float(np.mean(ridge_folds)), "ci": boot_ci(ridge_folds), "n": len(ridge_folds),
                           "definition": "5 per-fold values (deterministic), fold 3 excl. treatment"},
}

# ---------------------------------------------------------------- 3. sparsity
d = np.load(ROOT / "processed_data.npz")
X = np.concatenate([d["X_train"], d["X_val"], d["X_test"]])
zero_frac_processed = float((X == 0).mean())
dead_channels = float((np.abs(X).sum(axis=2) == 0).mean())

cache_dir = ROOT / "competitors" / "REFEvSet-V1" / "cache" / "beam_l1"
frac_patch_1ms, frac_roi_4ms, frac_roi_4ms_live = [], [], []
H_PATCH, W_PATCH = 45, 80  # 720/16, 1280/16
for f in sorted(cache_dir.glob("*.npz")):
    z = np.load(f, allow_pickle=True, mmap_mode="r")
    cnt = np.asarray(z["counts"])          # (2, 3600, T) uint8 @ 1 ms
    frac_patch_1ms.append(float((cnt == 0).mean()))
    # aggregate to our 3x3 ROI grid and ~4 ms bins
    T4 = cnt.shape[2] // 4
    c4 = cnt[:, :, : T4 * 4].reshape(2, 3600, T4, 4).sum(axis=3)
    rows = (np.arange(3600) // W_PATCH) * 3 // H_PATCH
    cols = (np.arange(3600) % W_PATCH) * 3 // W_PATCH
    roi = rows * 3 + cols
    roi9 = np.zeros((2, 9, T4), dtype=np.int64)
    for r9 in range(9):
        roi9[:, r9] = c4[:, roi == r9].sum(axis=1)
    frac_roi_4ms.append(float((roi9 == 0).mean()))
    live = roi9[:, roi9.reshape(2 * 9, -1).sum(1).reshape(2, 9).sum(0) > 0, :] \
        if False else roi9  # keep simple: live-channel variant below
    ch_live = roi9.reshape(18, T4)
    ch_live = ch_live[ch_live.sum(1) > 0]
    frac_roi_4ms_live.append(float((ch_live == 0).mean()))

out["sparsity"] = {
    "processed_X_exact_zero_frac": zero_frac_processed,
    "processed_X_dead_channel_frac": dead_channels,
    "raw_patch16px_1ms_zero_frac": [float(np.mean(frac_patch_1ms)), float(np.std(frac_patch_1ms))],
    "raw_roi3x3_4ms_zero_frac": [float(np.mean(frac_roi_4ms)), float(np.std(frac_roi_4ms))],
    "raw_roi3x3_4ms_zero_frac_live_channels": [float(np.mean(frac_roi_4ms_live)), float(np.std(frac_roi_4ms_live))],
    "notes": "patch-level from REF-EvSet level-1 caches (full recording span); "
             "ROI-grid 4 ms bins approximate our 3.906 ms pipeline bins",
}

# ---------------------------------------------- 4. dominant frequency per rec
md = common.reconstruct_row_metadata()
y = np.concatenate([d["y_train"], d["y_val"], d["y_test"]])
names_all = [r["name"] for r in md["train"] + md["val"] + md["test"]]
fs = 256.0
freqs = np.fft.rfftfreq(2048, 1 / fs)
band = (freqs >= 0.5) & (freqs <= 40)
f0s = {}
for i, n in enumerate(names_all):
    S = np.abs(np.fft.rfft(y[i] - y[i].mean()))
    f0s[n] = float(freqs[band][np.argmax(S[band])])
vals = np.array(sorted(f0s.values()))
anom = f0s["On_set3_trail1"]
others = np.array([v for k, v in f0s.items() if k != "On_set3_trail1"])
out["dominant_freq"] = {
    "On_set3_trail1_hz": anom,
    "others_mean_hz": float(others.mean()), "others_min_hz": float(others.min()),
    "others_max_hz": float(others.max()),
    "anomaly_offset_hz": float(anom - others.mean()),
    "per_recording": {k: round(v, 3) for k, v in sorted(f0s.items())},
}

common.save_json(out, HERE / "review_stats.json")

# ---------------------------------------------------------------- summary
print("=== 1. paired head-to-head (ours vs REF-EvSet, n=6) ===")
for k, p in paired.items():
    print(f"  {k:10s}: ours wins {p['ours_wins']}/6, sign p={p['sign_test_p']:.3f}, "
          f"wilcoxon p={p['wilcoxon_p'] if p['wilcoxon_p'] is None else round(p['wilcoxon_p'],3)}")
print("=== 2. bootstrap 95% CIs ===")
for k, v in out["bootstrap_ci_95"].items():
    if isinstance(v, dict):
        print(f"  {k}: mean {v['mean']:+.3f}, CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] (n={v['n']})")
print("=== 3. sparsity ===")
for k, v in out["sparsity"].items():
    print(f"  {k}: {v}")
print("=== 4. dominant frequency ===")
dfq = out["dominant_freq"]
print(f"  On_set3_trail1: {dfq['On_set3_trail1_hz']:.3f} Hz; others {dfq['others_mean_hz']:.3f} "
      f"[{dfq['others_min_hz']:.3f}, {dfq['others_max_hz']:.3f}] Hz")
