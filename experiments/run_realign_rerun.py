"""
Task 3b: re-run fixed-split training and CV fold 3 with the realigned
On_set2_trail3 (processed_data_realigned.npz), 5 seeds, warmup protocol.

Comparisons:
  fixed split : train = ALL 18 trials-1-3 recordings incl. realigned row 16
                vs the previous 17-sample (row dropped) protocol
  CV fold 3   : test = trial 3 incl. the realigned recording (train trials
                {1,2,5}, val trial 4) vs prior raw (diverged) and
                excluded-recording scores; per-recording R2 reported

Output: experiments/realign_rerun_results.json
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (agg, build_simple1dcnn, cv_folds, eval_all, predict,
                    reconstruct_row_metadata, save_json, train_model)

SEEDS = (0, 1, 2, 3, 4)
WARMUP = 5
NAME = "On_set2_trail3"


def load_realigned():
    d = np.load(common.ROOT / "processed_data_realigned.npz", allow_pickle=True)
    out = {f"{a}_{s}": d[f"{a}_{s}"].astype(np.float32)
           for a in ("X", "y") for s in ("train", "val", "test")}
    out["fs"] = out["y_train"].shape[1] / 8.0
    return out


def main():
    d = load_realigned()
    fs = d["fs"]
    results = {"seeds": list(SEEDS), "warmup_epochs": WARMUP}

    # ---------------- fixed split, all 18 train samples ----------------
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva, Xte, yte = d["X_val"], d["y_val"], d["X_test"], d["y_test"]
    per_test = []
    for seed in SEEDS:
        model, sc, _, ep = train_model(build_simple1dcnn, Xtr, ytr, Xva, yva,
                                       seed, warmup_epochs=WARMUP)
        mte = eval_all(yte, predict(model, sc, Xte), fs)
        per_test.append(mte)
        print(f"[fixed18 s{seed}] ep={ep:3d} test R2={mte['r2']:+.3f} "
              f"specNRMSE={mte['spec_nrmse']:.3f}")
    results["fixed_split_18train_realigned"] = {
        "test": agg(per_test), "test_per_seed": per_test,
        "n_collapsed": int(sum(m["r2"] < 0.1 for m in per_test)),
    }

    # ---------------- CV fold 3 ----------------
    md = reconstruct_row_metadata()
    X = np.concatenate([d["X_train"], d["X_val"], d["X_test"]], axis=0)
    y = np.concatenate([d["y_train"], d["y_val"], d["y_test"]], axis=0)
    rows = md["train"] + md["val"] + md["test"]
    trials = np.array([r["trial"] for r in rows])
    names = [r["name"] for r in rows]
    f3 = cv_folds(trials)[2]
    te_names = [names[i] for i in f3["test"]]
    bad = te_names.index(NAME)

    fold_full, fold_excl, per_rec_bad = [], [], []
    for seed in SEEDS:
        model, sc, _, ep = train_model(build_simple1dcnn, X[f3["train"]], y[f3["train"]],
                                       X[f3["val"]], y[f3["val"]], seed,
                                       warmup_epochs=WARMUP)
        p = predict(model, sc, X[f3["test"]])
        yte3 = y[f3["test"]]
        m_full = eval_all(yte3, p, fs)
        keep = [i for i in range(len(te_names)) if i != bad]
        m_excl = eval_all(yte3[keep], p[keep], fs)
        r2_bad = float(r2_score(yte3[bad], p[bad]))
        fold_full.append(m_full)
        fold_excl.append(m_excl)
        per_rec_bad.append(r2_bad)
        print(f"[fold3 s{seed}] ep={ep:3d} test R2 full={m_full['r2']:+.3f} "
              f"excl={m_excl['r2']:+.3f} | {NAME} alone: {r2_bad:+.3f}")

    results["cv_fold3_realigned"] = {
        "test_full6": agg(fold_full),
        "test_excl_bad": agg(fold_excl),
        "r2_on_realigned_recording": [float(np.mean(per_rec_bad)),
                                      float(np.std(per_rec_bad)),
                                      [round(v, 3) for v in per_rec_bad]],
    }
    save_json(results, HERE / "realign_rerun_results.json")

    r = results["fixed_split_18train_realigned"]["test"]["r2"]
    print(f"\nfixed(18 incl realigned): test R2 {r[0]:+.3f} ± {r[1]:.3f}  "
          f"(prev 17-sample: +0.636 ± 0.255 over 8 seeds)")
    rf = results["cv_fold3_realigned"]
    print(f"fold3 full-6:  {rf['test_full6']['r2'][0]:+.3f} ± {rf['test_full6']['r2'][1]:.3f}  "
          f"(prev raw: catastrophic, 6/8 < -1)")
    print(f"fold3 excl:    {rf['test_excl_bad']['r2'][0]:+.3f} ± {rf['test_excl_bad']['r2'][1]:.3f}  "
          f"(prev excl: +0.729 ± 0.258)")
    print(f"{NAME} alone:  {rf['r2_on_realigned_recording'][0]:+.3f} ± "
          f"{rf['r2_on_realigned_recording'][1]:.3f}  (prev: -68 to -572)")


if __name__ == "__main__":
    main()
