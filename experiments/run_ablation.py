"""
Post-pivot item 3: ROI-grid / polarity ablation on the fixed split (C2 evidence).

Input variants derived from processed_data.npz X (N, 18, 2048), whose channel
layout is [ROI0+, ROI0-, ROI1+, ROI1-, ...] (dataloader emits pos/neg per ROI):
  grid3x3_pol   : 18 ch — as-is (3x3 grid, polarity split)
  grid3x3_nopol : 9 ch  — pos+neg summed per ROI (spatial only)
  global_pol    : 2 ch  — summed over the 9 ROIs, polarity preserved
  global_nopol  : 1 ch  — fully collapsed control (no space, no polarity)

Same simple1dcnn (in_channels adjusted), warmup protocol, seeds (0,1,2),
train = trials 1-3 minus outlier row 16, selection on val, test once.

Note: X was per-sample abs-max normalized + per-channel mean-removed in the
dataloader, so channel sums are a proxy for a true coarser ROI binning —
stated in the report.

Output: experiments/ablation_results.json
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import (TRAIN_OUTLIER_ROW, agg, build_simple1dcnn, eval_all,
                    load_variant, predict, save_json, train_model)

SEEDS = (0, 1, 2)
WARMUP = 5


def reduce_channels(X, mode):
    if mode == "grid3x3_pol":
        return X
    if mode == "grid3x3_nopol":
        return X[:, 0::2] + X[:, 1::2]                      # (N, 9, T)
    if mode == "global_pol":
        return np.stack([X[:, 0::2].sum(1), X[:, 1::2].sum(1)], axis=1)  # (N, 2, T)
    if mode == "global_nopol":
        return X.sum(axis=1, keepdims=True)                  # (N, 1, T)
    raise ValueError(mode)


def main():
    d = load_variant("full")
    keep = np.array([i for i in range(18) if i != TRAIN_OUTLIER_ROW])
    fs = d["fs"]
    results = {"protocol": {"seeds": list(SEEDS), "warmup_epochs": WARMUP}, "variants": {}}

    for mode in ("grid3x3_pol", "grid3x3_nopol", "global_pol", "global_nopol"):
        Xtr = reduce_channels(d["X_train"], mode)[keep]
        ytr = d["y_train"][keep]
        Xva = reduce_channels(d["X_val"], mode)
        Xte = reduce_channels(d["X_test"], mode)
        yva, yte = d["y_val"], d["y_test"]
        C = Xtr.shape[1]
        build = partial(build_simple1dcnn, in_channels=C)

        per_val, per_test = [], []
        for seed in SEEDS:
            model, sc, _, ep = train_model(build, Xtr, ytr, Xva, yva, seed,
                                           warmup_epochs=WARMUP)
            mva = eval_all(yva, predict(model, sc, Xva), fs)
            mte = eval_all(yte, predict(model, sc, Xte), fs)
            per_val.append(mva)
            per_test.append(mte)
            print(f"[{mode:14s} C={C:2d} s{seed}] ep={ep:3d} "
                  f"val R2={mva['r2']:+.3f} test R2={mte['r2']:+.3f} "
                  f"specNRMSE={mte['spec_nrmse']:.3f}")

        results["variants"][mode] = {
            "in_channels": int(C),
            "n_params": common.n_params(build()),
            "val": agg(per_val), "test": agg(per_test),
            "val_per_seed": per_val, "test_per_seed": per_test,
        }

    save_json(results, common.ROOT / "experiments" / "ablation_results.json")
    print("\nSaved experiments/ablation_results.json")
    for mode, r in results["variants"].items():
        print(f"{mode:14s} C={r['in_channels']:2d}: val {r['val']['r2'][0]:+.3f}±{r['val']['r2'][1]:.3f} "
              f"test {r['test']['r2'][0]:+.3f}±{r['test']['r2'][1]:.3f}")


if __name__ == "__main__":
    main()
