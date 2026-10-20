"""
P0 item 2: 5-fold cross-validation by trial number.

Folds: test = trial k, val = trial (k % 5 + 1), train = remaining 3 trials
(every fold contains all 6 conditions in every split). Models: simple1dcnn
(full-resolution) and speck3k5 (square/32-output) x seeds (0,1,2), plus
ridge (alpha tuned on each fold's val) and the per-timestep train-mean.

All 30 recordings are used (the audit-2 outlier On_set2_trail3 is kept so
folds stay balanced; it sits in the train split of folds 1,2,4,5 and in the
test split of fold 3 — noted in the report).

Output: experiments/cv_results.json
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import (MODEL_BUILDERS, agg, cv_folds, eval_all, fit_ridge,
                    load_variant, predict, save_json, stack_all, train_model)

SEEDS = (0, 1, 2)
CV_MODELS = ("simple1dcnn", "speck3k5")


def main():
    results = {"models": {}, "baselines": {}}

    for name in CV_MODELS:
        build_fn, var = MODEL_BUILDERS[name]
        d = load_variant(var)
        X, y, trials, _ = stack_all(d)
        fs = d["fs"]
        folds = cv_folds(trials)

        test_metrics = []  # one dict per (fold, seed)
        per_fold = {}
        for f in folds:
            Xtr, ytr = X[f["train"]], y[f["train"]]
            Xva, yva = X[f["val"]], y[f["val"]]
            Xte, yte = X[f["test"]], y[f["test"]]
            fold_ms = []
            for seed in SEEDS:
                t0 = time.time()
                model, sc, _, ep = train_model(build_fn, Xtr, ytr, Xva, yva, seed)
                mte = eval_all(yte, predict(model, sc, Xte), fs)
                fold_ms.append(mte)
                test_metrics.append(mte)
                print(f"[{name} fold{f['k']} s{seed}] ep={ep:3d} "
                      f"test R2={mte['r2']:+.3f} specNRMSE={mte['spec_nrmse']:.3f} "
                      f"({time.time()-t0:.0f}s)")
            per_fold[f"fold{f['k']}"] = agg(fold_ms)

        results["models"][name] = {
            "variant": var, "fs": fs,
            "test_overall": agg(test_metrics),   # mean ± std over folds x seeds
            "per_fold": per_fold,
        }

    # ---------------- baselines per fold ----------------
    for var in ("full", "sq32"):
        d = load_variant(var)
        X, y, trials, _ = stack_all(d)
        fs = d["fs"]
        ridge_ms, mean_ms, alphas = [], [], []
        for f in cv_folds(trials):
            Xtr, ytr = X[f["train"]], y[f["train"]]
            Xva, yva = X[f["val"]], y[f["val"]]
            Xte, yte = X[f["test"]], y[f["test"]]
            m, _, alpha = fit_ridge(Xtr, ytr, Xva, yva)
            alphas.append(alpha)
            ridge_ms.append(eval_all(yte, m.predict(Xte.reshape(len(Xte), -1)), fs))
            ybar = ytr.mean(axis=0)
            mean_ms.append(eval_all(yte, np.tile(ybar, (len(yte), 1)), fs))
            print(f"[baselines {var} fold{f['k']}] ridge(a={alpha:g}) "
                  f"test R2={ridge_ms[-1]['r2']:+.3f} | mean-sig R2={mean_ms[-1]['r2']:+.3f}")
        results["baselines"][var] = {
            "ridge": {"test_overall": agg(ridge_ms), "alphas": alphas,
                      "per_fold": [m["r2"] for m in ridge_ms]},
            "per_timestep_mean": {"test_overall": agg(mean_ms),
                                  "per_fold": [m["r2"] for m in mean_ms]},
        }

    save_json(results, common.ROOT / "experiments" / "cv_results.json")
    print("\nSaved experiments/cv_results.json")

    for name in CV_MODELS:
        r = results["models"][name]["test_overall"]["r2"]
        print(f"{name}: CV test R2 = {r[0]:+.3f} ± {r[1]:.3f}")
    for var in ("full", "sq32"):
        r = results["baselines"][var]["ridge"]["test_overall"]["r2"]
        print(f"ridge({var}): CV test R2 = {r[0]:+.3f} ± {r[1]:.3f}")


if __name__ == "__main__":
    main()
