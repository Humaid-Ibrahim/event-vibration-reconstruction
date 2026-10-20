"""
Headline promotion: full unified protocol on the 10x10 ROI-grid representation
(processed_data_grid10.npz, 200 ch), mirroring run_seed_protocol.py.

1. headline split (test=2, val=4, train=1/3/5 minus On_set2_trail3), 10 slots
2. secondary split (test=5, val=4, train=1/2/3 minus excluded), 10 slots
3. 5-fold CV, seeds 42-44, fold-3 scored excluding On_set2_trail3
4. ridge + per-timestep mean on the 10x10 representation (per split / per fold)
5. REF-EvSet head-to-head consistency metrics (secondary best-val seed,
   6 trial-5 recordings, 6 s onset frame, our metric defs, no fitting)
6. prediction files for figures: predictions_simple1dcnn_10x10.npz (headline)

Output: experiments/headline_10x10_results.json
"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (agg, build_simple1dcnn, cv_folds, eval_all, fit_ridge,
                    predict, save_json)
from run_seed_protocol import (train_with_restart, boot_ci, SEEDS10, SEEDS3,  # noqa: E402
                               EXCLUDE)
from run_refevset_comparison import onset_window, our_metrics  # noqa: E402

FS = 256.0
BUILD = partial(build_simple1dcnn, in_channels=200)


def load10():
    d = np.load(ROOT / "processed_data_grid10.npz")
    X = np.concatenate([d["X_train"], d["X_val"], d["X_test"]]).astype(np.float32)
    y = np.concatenate([d["y_train"], d["y_val"], d["y_test"]]).astype(np.float32)
    md = common.reconstruct_row_metadata()
    rows = md["train"] + md["val"] + md["test"]
    return X, y, np.array([r["trial"] for r in rows]), [r["name"] for r in rows], md


def split_masks(trials, names, test_trial, train_trials):
    test_m = trials == test_trial
    val_m = trials == 4
    train_m = np.array([(t in train_trials) and n != EXCLUDE
                        for t, n in zip(trials, names)])
    assert not (train_m & (test_m | val_m)).any()
    return train_m, val_m, test_m


def run_split(X, y, trials, names, test_trial, train_trials, label):
    train_m, val_m, test_m = split_masks(trials, names, test_trial, train_trials)
    Xtr, ytr = X[train_m], y[train_m]
    Xva, yva = X[val_m], y[val_m]
    Xte, yte = X[test_m], y[test_m]

    per_test, infos = [], []
    best = (None, -np.inf, None)  # (slot, val_r2, pred_te)
    for seed in SEEDS10:
        model, sc, info = train_with_restart(BUILD, Xtr, ytr, Xva, yva, seed,
                                             log_prefix=f"[{label}]")
        pte = predict(model, sc, Xte)
        mte = eval_all(yte, pte, FS)
        per_test.append(mte)
        infos.append(info)
        print(f"[{label} slot {seed}] val={info['final_val_r2']:+.3f} "
              f"test R2={mte['r2']:+.3f} restarts={info['n_restarts']}"
              f"{' UNRESOLVED' if info['unresolved'] else ''}", flush=True)
        if info["final_val_r2"] > best[1]:
            best = (seed, info["final_val_r2"], pte)

    r2s = [m["r2"] for m in per_test]
    res = {"test": agg(per_test), "test_per_slot": per_test, "slots": infos,
           "n_slots_restarted": int(sum(i["n_restarts"] > 0 for i in infos)),
           "n_unresolved": int(sum(i["unresolved"] for i in infos)),
           "test_r2_ci95": boot_ci(r2s), "best_val_slot": best[0]}

    ybar = ytr.mean(axis=0)
    m, _, alpha = fit_ridge(Xtr, ytr, Xva, yva)
    res["baselines"] = {
        "per_timestep_mean": {"test": eval_all(yte, np.tile(ybar, (len(yte), 1)), FS)},
        "ridge10x10": {"alpha": alpha,
                       "test": eval_all(yte, m.predict(Xte.reshape(len(Xte), -1)), FS)},
    }
    return res, best[2], yte


def main():
    X, y, trials, names, md = load10()
    results = {"representation": "10x10 ROI grid, polarity split (200 ch), "
                                 "simple1dcnn in_channels=200 (139,704 params)",
               "protocol": "seeds 42-51 (splits) / 42-44 (CV); restart-on-collapse "
                           "val R2 < 0.3 (max 3 attempts); warmup 5 -> 5e-4 cosine; "
                           "ES patience 15; val-only selection"}

    print("=== headline split (test=2) ===")
    res_h, pred_h, yte_h = run_split(X, y, trials, names, 2, (1, 3, 5), "headline10")
    results["headline_trial2"] = res_h
    np.savez_compressed(HERE / "predictions_simple1dcnn_10x10.npz",
                        y_test=yte_h, y_test_pred=pred_h, fs=FS,
                        best_slot=res_h["best_val_slot"])

    print("\n=== secondary split (test=5) ===")
    res_s, pred_s, yte_s = run_split(X, y, trials, names, 5, (1, 2, 3), "secondary10")
    results["secondary_trial5"] = res_s

    # ---------------- CV ----------------
    print("\n=== 5-fold CV ===")
    cv_runs, per_fold, cv_infos, ridge_folds = [], {}, [], []
    for f in cv_folds(trials):
        te_names = [names[i] for i in f["test"]]
        excl = te_names.index(EXCLUDE) if EXCLUDE in te_names else None
        keep = [i for i in range(len(te_names)) if i != excl]
        fold_ms = []
        for seed in SEEDS3:
            model, sc, info = train_with_restart(
                BUILD, X[f["train"]], y[f["train"]], X[f["val"]], y[f["val"]],
                seed, log_prefix=f"[cv10 f{f['k']}]")
            p = predict(model, sc, X[f["test"]])
            yte = y[f["test"]]
            mte = eval_all(yte[keep], p[keep], FS) if excl is not None \
                else eval_all(yte, p, FS)
            fold_ms.append(mte)
            cv_runs.append(mte)
            info["fold"] = f["k"]
            cv_infos.append(info)
            print(f"[cv10 fold{f['k']} slot {seed}] test R2={mte['r2']:+.3f} "
                  f"restarts={info['n_restarts']}"
                  f"{' UNRESOLVED' if info['unresolved'] else ''}", flush=True)
        per_fold[f"fold{f['k']}"] = agg(fold_ms)
        mR, _, alpha = fit_ridge(X[f["train"]], y[f["train"]], X[f["val"]], y[f["val"]])
        pR = mR.predict(X[f["test"]].reshape(len(f["test"]), -1))
        yte = y[f["test"]]
        mRm = eval_all(yte[keep], pR[keep], FS) if excl is not None \
            else eval_all(yte, pR, FS)
        mRm["alpha"] = alpha
        ridge_folds.append(mRm)
        print(f"[cv10 fold{f['k']} ridge] test R2={mRm['r2']:+.3f}")

    r2s = [m["r2"] for m in cv_runs]
    results["cv"] = {
        "test_overall": agg(cv_runs), "per_fold": per_fold,
        "r2_ci95": boot_ci(r2s), "slots": cv_infos,
        "n_slots_restarted": int(sum(i["n_restarts"] > 0 for i in cv_infos)),
        "n_unresolved": int(sum(i["unresolved"] for i in cv_infos)),
        "ridge10x10": {"per_fold_r2": [m["r2"] for m in ridge_folds],
                       "test_overall": agg([{k: v for k, v in m.items() if k != "alpha"}
                                            for m in ridge_folds])},
        "note": "fold 3 scored excluding On_set2_trail3 (both methods)",
    }

    # ---------------- REF-EvSet consistency (secondary best-val slot) --------
    te_names = [r["name"] for r in md["test"]]  # trial-5 rows
    ours_rows = []
    for i, name in enumerate(te_names):
        sl = onset_window(yte_s[i])
        m = our_metrics(yte_s[i][sl], pred_s[i][sl])
        m["name"] = name
        ours_rows.append(m)
    keys = ("r2", "spec_nrmse", "decay_err", "env_r2")
    results["refevset_frame_ours10x10"] = {
        "per_recording": ours_rows,
        "agg": {k: [float(np.mean([r[k] for r in ours_rows])),
                    float(np.std([r[k] for r in ours_rows]))] for k in keys},
        "note": "secondary-split best-val slot, 6 s onset frame, no lag/scale "
                "fitting; REF-EvSet rows unchanged (refevset_comparison.json)",
    }

    save_json(results, HERE / "headline_10x10_results.json")

    print("\n=== SUMMARY (10x10) ===")
    for k in ("headline_trial2", "secondary_trial5"):
        r = results[k]
        t = r["test"]
        print(f"{k}: R2 {t['r2'][0]:+.3f}±{t['r2'][1]:.3f} CI "
              f"{np.round(r['test_r2_ci95'],3).tolist()} | spec {t['spec_nrmse'][0]:.3f} "
              f"| env {t['env_r2'][0]:.3f} | dbl {t['dbl_ratio_pred'][0]:.3f} "
              f"(true {t['dbl_ratio_true'][0]:.3f}) | restarts {r['n_slots_restarted']}/10 "
              f"| ridge10 {r['baselines']['ridge10x10']['test']['r2']:+.3f}")
    c = results["cv"]
    print(f"cv: R2 {c['test_overall']['r2'][0]:+.3f}±{c['test_overall']['r2'][1]:.3f} "
          f"CI {np.round(c['r2_ci95'],3).tolist()} vs ridge10 "
          f"{c['ridge10x10']['test_overall']['r2'][0]:+.3f} | restarts "
          f"{c['n_slots_restarted']}/15 unresolved {c['n_unresolved']}")
    a = results["refevset_frame_ours10x10"]["agg"]
    print(f"head-to-head frame: R2 {a['r2'][0]:+.3f}±{a['r2'][1]:.3f} "
          f"spec {a['spec_nrmse'][0]:.3f} env {a['env_r2'][0]:+.3f}")


if __name__ == "__main__":
    main()
