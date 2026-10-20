"""
Post-pivot item 1 + 2: seed stability with LR warmup for simple1dcnn.

1. Fixed split (train = trials 1-3 minus outlier row 16, val = 4, test = 5),
   8 seeds (0-7), linear warmup 5 epochs -> 5e-4 then cosine decay.
   Collapses flagged (test R² < 0.1). Compared against P0's no-warmup runs.
2. 5-fold CV by trial with the same 8-seed warmup protocol + per-fold ridge.

Outputs: experiments/stability_results.json, experiments/cv_warmup_results.json,
         experiments/predictions_simple1dcnn.npz (refreshed: best-val warmup seed)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import (TRAIN_OUTLIER_ROW, agg, build_simple1dcnn, cv_folds,
                    eval_all, fit_ridge, load_variant, predict, save_json,
                    stack_all, train_model)

SEEDS = tuple(range(8))
WARMUP = 5


def fixed_split():
    d = load_variant("full")
    keep = np.array([i for i in range(18) if i != TRAIN_OUTLIER_ROW])
    Xtr, ytr = d["X_train"][keep], d["y_train"][keep]
    Xva, yva, Xte, yte = d["X_val"], d["y_val"], d["X_test"], d["y_test"]
    fs = d["fs"]

    per_val, per_test, infos = [], [], []
    best = (None, -np.inf, None, None)
    for seed in SEEDS:
        t0 = time.time()
        model, sc, _, ep = train_model(build_simple1dcnn, Xtr, ytr, Xva, yva,
                                       seed, warmup_epochs=WARMUP)
        pva, pte = predict(model, sc, Xva), predict(model, sc, Xte)
        mva, mte = eval_all(yva, pva, fs), eval_all(yte, pte, fs)
        per_val.append(mva)
        per_test.append(mte)
        collapsed = mte["r2"] < 0.1
        infos.append({"seed": seed, "epochs_run": ep, "collapsed": bool(collapsed)})
        print(f"[fixed s{seed}] ep={ep:3d} val R2={mva['r2']:+.3f} test R2={mte['r2']:+.3f} "
              f"specNRMSE={mte['spec_nrmse']:.3f} {'COLLAPSED' if collapsed else ''} "
              f"({time.time()-t0:.0f}s)")
        if mva["r2"] > best[1]:
            best = (seed, mva["r2"], pva, pte)

    res = {
        "protocol": {"seeds": list(SEEDS), "warmup_epochs": WARMUP,
                     "lr": 5e-4, "schedule": "linear warmup + cosine"},
        "seeds": infos,
        "val": agg(per_val), "test": agg(per_test),
        "val_per_seed": per_val, "test_per_seed": per_test,
        "n_collapsed": int(sum(i["collapsed"] for i in infos)),
        "best_seed": best[0],
    }
    np.savez_compressed(common.ROOT / "experiments" / "predictions_simple1dcnn.npz",
                        y_val=yva, y_val_pred=best[2], y_test=yte, y_test_pred=best[3],
                        best_seed=best[0], fs=fs)
    save_json(res, common.ROOT / "experiments" / "stability_results.json")
    r2s = [m["r2"] for m in per_test]
    print(f"\nfixed split: test R2 {np.mean(r2s):+.3f} ± {np.std(r2s):.3f} "
          f"(min {min(r2s):+.3f}, max {max(r2s):+.3f}), "
          f"{res['n_collapsed']}/{len(SEEDS)} collapsed")
    return res


def cv_warmup():
    d = load_variant("full")
    X, y, trials, _ = stack_all(d)
    fs = d["fs"]

    model_runs, per_fold = [], {}
    ridge_ms = []
    for f in cv_folds(trials):
        Xtr, ytr = X[f["train"]], y[f["train"]]
        Xva, yva = X[f["val"]], y[f["val"]]
        Xte, yte = X[f["test"]], y[f["test"]]
        fold_ms = []
        for seed in SEEDS:
            model, sc, _, ep = train_model(build_simple1dcnn, Xtr, ytr, Xva, yva,
                                           seed, warmup_epochs=WARMUP)
            mte = eval_all(yte, predict(model, sc, Xte), fs)
            mte["_fold"], mte["_seed"] = f["k"], seed
            fold_ms.append(mte)
            model_runs.append(mte)
            print(f"[cv fold{f['k']} s{seed}] ep={ep:3d} test R2={mte['r2']:+.3f} "
                  f"{'<<DIVERGED' if mte['r2'] < -1 else ''}")
        per_fold[f"fold{f['k']}"] = agg([{k: v for k, v in m.items()
                                          if not k.startswith('_')} for m in fold_ms])
        m, _, alpha = fit_ridge(Xtr, ytr, Xva, yva)
        rm = eval_all(yte, m.predict(Xte.reshape(len(Xte), -1)), fs)
        rm["alpha"] = alpha
        ridge_ms.append(rm)
        print(f"[cv fold{f['k']} ridge] test R2={rm['r2']:+.3f} (a={alpha:g})")

    clean = [{k: v for k, v in m.items() if not k.startswith('_')} for m in model_runs]
    r2s = np.array([m["r2"] for m in model_runs])
    res = {
        "protocol": {"seeds": list(SEEDS), "warmup_epochs": WARMUP},
        "simple1dcnn": {
            "test_overall": agg(clean),
            "per_fold": per_fold,
            "r2_all_runs": [float(r) for r in r2s],
            "r2_median": float(np.median(r2s)),
            "n_diverged_lt_-1": int((r2s < -1).sum()),
            "n_collapsed_lt_0.1": int((r2s < 0.1).sum()),
        },
        "ridge": {
            "test_overall": agg([{k: v for k, v in m.items() if k != 'alpha'}
                                 for m in ridge_ms]),
            "per_fold_r2": [m["r2"] for m in ridge_ms],
            "alphas": [m["alpha"] for m in ridge_ms],
        },
    }
    save_json(res, common.ROOT / "experiments" / "cv_warmup_results.json")
    print(f"\nCV: simple1dcnn R2 {r2s.mean():+.3f} ± {r2s.std():.3f} "
          f"(median {np.median(r2s):+.3f}; diverged<-1: {(r2s<-1).sum()}), "
          f"ridge {np.mean([m['r2'] for m in ridge_ms]):+.3f}")
    return res


if __name__ == "__main__":
    print("=== fixed split, 8 seeds, warmup ===")
    fixed_split()
    print("\n=== 5-fold CV, 8 seeds, warmup ===")
    cv_warmup()
