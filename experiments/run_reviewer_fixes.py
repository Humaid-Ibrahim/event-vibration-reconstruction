"""
Reviewer fixes (SAFEPROCESS 2027 revision).

Task 1: n=30 REF-EvSet comparison. Every recording gets a model prediction
from the CV fold where it was a TEST item (10x10 representation, unified
protocol, best-val run per fold). Paired per-recording stats vs REF-EvSet's
training-free reconstruction (rows already in refevset_comparison.json,
scored with the same our_metrics on the same 6 s onset frame).

Task 2: decay-rate diagnosis on the main split (test=trial 2):
(a) signed decay errors model/ridge (bias vs noise);
(b) smoothing ablation (Savitzky-Golay on envelope; moving average on signal).

Output: experiments/reviewer_fixes_results.json, cv_predictions_10x10.npz
"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import build_simple1dcnn, cv_folds, fit_ridge, predict, save_json  # noqa: E402
from run_seed_protocol import train_with_restart, SEEDS3, EXCLUDE  # noqa: E402
from run_headline_10x10 import load10, split_masks  # noqa: E402
from run_refevset_comparison import onset_window, our_metrics  # noqa: E402

FS = 256.0
BUILD = partial(build_simple1dcnn, in_channels=200)
KEYS = ("r2", "spec_nrmse", "decay_err", "env_r2")


def norm_name(n):
    return n.replace("trai4", "trail4")


# ---------------------------------------------------------------------------
# Task 1: CV predictions for all 30 recordings, paired vs REF-EvSet
# ---------------------------------------------------------------------------

def task1():
    X, y, trials, names, md = load10()
    preds = np.zeros_like(y)
    fold_of = np.zeros(len(y), dtype=int)

    for f in cv_folds(trials):
        best = (None, -np.inf, None, None)  # (seed, val_r2, model, sc)
        for seed in SEEDS3:
            model, sc, info = train_with_restart(
                BUILD, X[f["train"]], y[f["train"]], X[f["val"]], y[f["val"]],
                seed, log_prefix=f"[rf cv f{f['k']}]")
            if info["final_val_r2"] > best[1]:
                best = (seed, info["final_val_r2"], model, sc)
        p = predict(best[2], best[3], X[f["test"]])
        preds[f["test"]] = p
        fold_of[f["test"]] = f["k"]
        print(f"[fold {f['k']}] best seed {best[0]} val R2 {best[1]:+.3f}", flush=True)

    np.savez_compressed(HERE / "cv_predictions_10x10.npz",
                        y=y, preds=preds, names=np.array(names),
                        fold=fold_of, fs=FS)

    # per-recording metrics on the 6 s onset frame
    ours = {}
    for i, name in enumerate(names):
        sl = onset_window(y[i])
        m = our_metrics(y[i][sl], preds[i][sl])
        ours[norm_name(name)] = m

    ref = json.load(open(HERE / "refevset_comparison.json"))["per_recording"]["refevset"]
    ref = {norm_name(r["name"]): r for r in ref}

    rows = []
    for name in sorted(ours):
        if name not in ref:
            print(f"WARNING: {name} missing from REF rows")
            continue
        rows.append({
            "name": name,
            "ours_r2": ours[name]["r2"], "ref_r2": ref[name]["r2"],
            "ours_nrmse": ours[name]["spec_nrmse"], "ref_nrmse": ref[name]["spec_nrmse"],
        })

    def stats(rs):
        o_r2 = np.array([r["ours_r2"] for r in rs])
        f_r2 = np.array([r["ref_r2"] for r in rs])
        o_nr = np.array([r["ours_nrmse"] for r in rs])
        f_nr = np.array([r["ref_nrmse"] for r in rs])
        n = len(rs)
        w_r2 = int((o_r2 > f_r2).sum())
        w_nr = int((o_nr < f_nr).sum())
        return {
            "n": n,
            "ours_r2_mean": [float(o_r2.mean()), float(o_r2.std())],
            "ref_r2_mean": [float(f_r2.mean()), float(f_r2.std())],
            "ours_nrmse_mean": [float(o_nr.mean()), float(o_nr.std())],
            "ref_nrmse_mean": [float(f_nr.mean()), float(f_nr.std())],
            "wins_r2": w_r2, "wins_nrmse": w_nr,
            "sign_p_r2": float(binomtest(w_r2, n, 0.5).pvalue),
            "sign_p_nrmse": float(binomtest(w_nr, n, 0.5).pvalue),
            "wilcoxon_p_r2": float(wilcoxon(o_r2, f_r2).pvalue),
            "wilcoxon_p_nrmse": float(wilcoxon(o_nr, f_nr).pvalue),
        }

    return {
        "frame": "6 s onset window @256 Hz; REF-EvSet rows carry their "
                 "per-recording lag+scale fit (favorable to them); CV-fold "
                 "predictions for ours, no fitting",
        "per_recording": rows,
        "stats_n30": stats(rows),
        "stats_n29_excl": stats([r for r in rows if r["name"] != EXCLUDE]),
    }


# ---------------------------------------------------------------------------
# Task 2: decay diagnostics on the main split
# ---------------------------------------------------------------------------

def task2():
    X, y, trials, names, md = load10()
    train_m, val_m, test_m = split_masks(trials, names, 2, (1, 3, 5))

    d = np.load(HERE / "predictions_simple1dcnn_10x10.npz")
    yte, pte = d["y_test"], d["y_test_pred"]

    mR, _, alpha = fit_ridge(X[train_m], y[train_m], X[val_m], y[val_m])
    pR = mR.predict(X[test_m].reshape(test_m.sum(), -1))

    # (a) signed decay errors
    st = common.decay_slope(yte, FS)
    sm = common.decay_slope(pte, FS)
    sr = common.decay_slope(pR, FS)
    signed = {
        "model": {"mean": float((sm - st).mean()), "std": float((sm - st).std()),
                  "per_rec": (sm - st).tolist()},
        "ridge": {"mean": float((sr - st).mean()), "std": float((sr - st).std()),
                  "per_rec": (sr - st).tolist()},
        "truth_decay": st.tolist(),
    }

    # (b) smoothing ablation on the model prediction
    def metrics_variant(pred):
        out = {"r2": float(r2_score(yte.ravel(), pred.ravel())),
               "spec_nrmse": float(common.spectral_nrmse(yte, pred, FS).mean())}
        sp = common.decay_slope(pred, FS)
        out["decay_err"] = float(np.abs(sp - st).mean())
        et = common.smooth_env(yte, FS)
        ep = common.smooth_env(pred, FS)
        out["env_r2"] = float(r2_score(et.ravel(), ep.ravel()))
        return out

    def ma(x, k):
        kern = np.ones(k) / k
        return np.apply_along_axis(lambda r: np.convolve(r, kern, "same"), 1, x)

    variants = {"raw": metrics_variant(pte),
                "ma3": metrics_variant(ma(pte, 3)),
                "ma5": metrics_variant(ma(pte, 5))}

    # Savitzky-Golay applied to the envelope only affects decay; recompute
    # decay from the filtered envelope, other metrics unchanged from raw
    for po in (2, 3):
        env = np.abs(np.apply_along_axis(
            lambda r: savgol_filter(r, 11, po), 1,
            np.abs(__import__("scipy.signal", fromlist=["hilbert"]).hilbert(pte, axis=-1))))
        t = np.arange(env.shape[1]) / FS
        logs = np.log(np.maximum(env, 1e-8))
        A = np.vstack([t, np.ones_like(t)]).T
        slopes = np.array([np.linalg.lstsq(A, row, rcond=None)[0][0] for row in logs])
        v = dict(variants["raw"])
        v["decay_err"] = float(np.abs(slopes - st).mean())
        v["note"] = "envelope-only filter; r2/nrmse/env_r2 = raw"
        variants[f"savgol_w11_p{po}"] = v

    return {"split": "test=trial2, ridge alpha " + str(alpha),
            "signed_decay_error": signed, "smoothing": variants}


def main():
    res = {"task1_n30_comparison": task1(), "task2_decay": task2()}
    save_json(res, HERE / "reviewer_fixes_results.json")

    t1 = res["task1_n30_comparison"]
    for k in ("stats_n30", "stats_n29_excl"):
        s = t1[k]
        print(f"\n{k}: n={s['n']} | ours R2 {s['ours_r2_mean'][0]:+.3f} vs ref "
              f"{s['ref_r2_mean'][0]:+.3f} | wins {s['wins_r2']}/{s['n']} "
              f"sign p={s['sign_p_r2']:.4f} wilcoxon p={s['wilcoxon_p_r2']:.4f} | "
              f"NRMSE wins {s['wins_nrmse']}/{s['n']} sign p={s['sign_p_nrmse']:.4f}")
    t2 = res["task2_decay"]
    sd = t2["signed_decay_error"]
    print(f"\nsigned decay err: model {sd['model']['mean']:+.3f}±{sd['model']['std']:.3f} "
          f"| ridge {sd['ridge']['mean']:+.3f}±{sd['ridge']['std']:.3f}")
    for v, m in t2["smoothing"].items():
        print(f"{v:16s} decay {m['decay_err']:.3f} r2 {m['r2']:+.3f} "
              f"nrmse {m['spec_nrmse']:.3f} env {m['env_r2']:+.3f}")


if __name__ == "__main__":
    main()
