"""
P0 item 1 + 3 + 4: multi-epoch training on the fixed split (train = trials
1-3 minus the audit-2 outlier row 16 / On_set2_trail3, val = trial 4,
test = trial 5), 3 seeds per model, baselines refit on the same reduced
train set, frequency-domain metrics for everything.

Outputs:
  experiments/fixed_split_results.json
  experiments/predictions_<model>.npz  (best-val seed per model)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import (MODEL_BUILDERS, TRAIN_OUTLIER_ROW, agg, eval_all,
                    fit_ridge, load_variant, predict, save_json, train_model)

SEEDS = (0, 1, 2)


def main():
    results = {"device": str(common.DEVICE), "dropped_train_row": TRAIN_OUTLIER_ROW,
               "dropped_recording": "On_set2_trail3", "models": {}, "baselines": {}}

    variants = {v: load_variant(v) for v in ("full", "y128", "sq32")}
    keep = np.array([i for i in range(18) if i != TRAIN_OUTLIER_ROW])

    # ---------------- learned models ----------------
    for name, (build_fn, var) in MODEL_BUILDERS.items():
        d = variants[var]
        Xtr, ytr = d["X_train"][keep], d["y_train"][keep]
        Xva, yva, Xte, yte = d["X_val"], d["y_val"], d["X_test"], d["y_test"]
        fs = d["fs"]

        per_seed_val, per_seed_test, infos = [], [], []
        best = (None, -np.inf, None, None)  # (seed, val_r2, pred_va, pred_te)
        for seed in SEEDS:
            t0 = time.time()
            model, sc, best_vl, ep = train_model(build_fn, Xtr, ytr, Xva, yva, seed)
            pva = predict(model, sc, Xva)
            pte = predict(model, sc, Xte)
            mva = eval_all(yva, pva, fs)
            mte = eval_all(yte, pte, fs)
            per_seed_val.append(mva)
            per_seed_test.append(mte)
            infos.append({"seed": seed, "epochs_run": ep, "best_val_loss": best_vl,
                          "train_s": round(time.time() - t0, 1)})
            print(f"[{name} s{seed}] ep={ep:3d} val R2={mva['r2']:+.3f} "
                  f"test R2={mte['r2']:+.3f} specNRMSE(te)={mte['spec_nrmse']:.3f} "
                  f"({infos[-1]['train_s']}s)")
            if mva["r2"] > best[1]:
                best = (seed, mva["r2"], pva, pte)

        results["models"][name] = {
            "n_params": common.n_params(build_fn()),
            "variant": var, "fs": fs, "T_out": d["T_out"],
            "seeds": infos,
            "val": agg(per_seed_val), "test": agg(per_seed_test),
            "val_per_seed": per_seed_val, "test_per_seed": per_seed_test,
            "best_seed": best[0],
        }
        np.savez_compressed(common.ROOT / "experiments" / f"predictions_{name}.npz",
                            y_val=yva, y_val_pred=best[2],
                            y_test=yte, y_test_pred=best[3],
                            best_seed=best[0], fs=fs)

    # ---------------- baselines (same reduced train set) ----------------
    for var in ("full", "y128", "sq32"):
        d = variants[var]
        Xtr, ytr = d["X_train"][keep], d["y_train"][keep]
        Xva, yva, Xte, yte = d["X_val"], d["y_val"], d["X_test"], d["y_test"]
        fs = d["fs"]

        ybar = ytr.mean(axis=0)
        bl = {
            "per_timestep_mean": {
                "val": eval_all(yva, np.tile(ybar, (len(yva), 1)), fs),
                "test": eval_all(yte, np.tile(ybar, (len(yte), 1)), fs),
            }
        }
        m, val_r2, alpha = fit_ridge(Xtr, ytr, Xva, yva)
        bl["ridge"] = {
            "alpha": alpha,
            "val": eval_all(yva, m.predict(Xva.reshape(len(Xva), -1)), fs),
            "test": eval_all(yte, m.predict(Xte.reshape(len(Xte), -1)), fs),
        }
        results["baselines"][var] = bl
        print(f"[baselines {var}] ridge(a={alpha:g}) val R2={bl['ridge']['val']['r2']:+.3f} "
              f"test R2={bl['ridge']['test']['r2']:+.3f} | "
              f"mean-sig val R2={bl['per_timestep_mean']['val']['r2']:+.3f} "
              f"test R2={bl['per_timestep_mean']['test']['r2']:+.3f}")

    save_json(results, common.ROOT / "experiments" / "fixed_split_results.json")
    print("\nSaved experiments/fixed_split_results.json")


if __name__ == "__main__":
    main()
