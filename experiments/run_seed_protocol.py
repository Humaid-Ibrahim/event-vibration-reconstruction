"""
Seed-hygiene re-run of every paper number (seeds 42-51 / 42-44, restart-on-collapse).

Protocol:
- Seeds 42..51 (headline + secondary split), 42..44 (CV, ablation).
- Restart-on-collapse: a run is collapsed if its VALIDATION R2 at early stop is
  below VAL_COLLAPSE_THR. Collapsed slots restart with seed+1000, then +2000
  (max 3 attempts). Detection uses validation only - test is never touched.
  NOTE: threshold is 0.3, not the nominal 0.1 - the mean-predictor collapse
  mode itself achieves val R2 ~ 0.19-0.21 on these phase-locked trials
  (healthy runs reach 0.74-0.85), so 0.1 would never trigger. Runs below 0.1
  are counted separately (expected: none).
- If all 3 attempts collapse, the best-val attempt is kept and flagged
  "unresolved" (reported, not hidden).

Outputs: experiments/seed_protocol_results.json
"""
import sys
from pathlib import Path

import numpy as np
from functools import partial
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (agg, build_simple1dcnn, cv_folds, eval_all, fit_ridge,
                    load_variant, predict, save_json, stack_all, train_model)

SEEDS10 = tuple(range(42, 52))
SEEDS3 = (42, 43, 44)
WARMUP = 5
VAL_COLLAPSE_THR = 0.3
NOMINAL_THR = 0.1
EXCLUDE = "On_set2_trail3"
rng = np.random.default_rng(0)


def boot_ci(x, n_boot=10_000):
    x = np.asarray(x, dtype=float)
    bs = rng.choice(x, size=(n_boot, len(x))).mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def train_with_restart(build_fn, Xtr, ytr, Xva, yva, seed, log_prefix=""):
    """Returns (model, sc, info). Restart rule: val R2 < VAL_COLLAPSE_THR."""
    attempts = []
    best = None
    for k, s in enumerate((seed, seed + 1000, seed + 2000)):
        model, sc, _, ep = train_model(build_fn, Xtr, ytr, Xva, yva, s,
                                       warmup_epochs=WARMUP)
        val_r2 = float(r2_score(yva.ravel(), predict(model, sc, Xva).ravel()))
        attempts.append({"seed": s, "epochs_run": ep, "val_r2": round(val_r2, 4)})
        if best is None or val_r2 > best[2]:
            best = (model, sc, val_r2)
        if val_r2 >= VAL_COLLAPSE_THR:
            info = {"slot_seed": seed, "attempts": attempts, "n_restarts": k,
                    "final_seed": s, "final_val_r2": round(val_r2, 4),
                    "unresolved": False,
                    "below_nominal_0.1": bool(val_r2 < NOMINAL_THR)}
            if k:
                print(f"{log_prefix}  slot {seed}: {k} restart(s) -> seed {s}")
            return model, sc, info
        print(f"{log_prefix}  slot {seed}: attempt seed {s} COLLAPSED (val R2 {val_r2:+.3f})")
    info = {"slot_seed": seed, "attempts": attempts, "n_restarts": 2,
            "final_seed": attempts[int(np.argmax([a['val_r2'] for a in attempts]))]["seed"],
            "final_val_r2": round(best[2], 4), "unresolved": True,
            "below_nominal_0.1": bool(best[2] < NOMINAL_THR)}
    print(f"{log_prefix}  slot {seed}: UNRESOLVED after 3 attempts (best val {best[2]:+.3f})")
    return best[0], best[1], info


def run_split(X, y, trials, names, test_trial, val_trial, train_trials, label):
    test_m = trials == test_trial
    val_m = trials == val_trial
    train_m = np.array([(t in train_trials) and n != EXCLUDE
                        for t, n in zip(trials, names)])
    assert not (train_m & (test_m | val_m)).any()
    Xtr, ytr = X[train_m], y[train_m]
    Xva, yva = X[val_m], y[val_m]
    Xte, yte = X[test_m], y[test_m]

    per_test, infos = [], []
    for seed in SEEDS10:
        model, sc, info = train_with_restart(build_simple1dcnn, Xtr, ytr, Xva, yva,
                                             seed, log_prefix=f"[{label}]")
        mte = eval_all(yte, predict(model, sc, Xte), 256.0)
        per_test.append(mte)
        infos.append(info)
        print(f"[{label} slot {seed}] val R2={info['final_val_r2']:+.3f} "
              f"test R2={mte['r2']:+.3f} restarts={info['n_restarts']}")

    r2s = [m["r2"] for m in per_test]
    res = {
        "test": agg(per_test), "test_per_slot": per_test, "slots": infos,
        "n_slots_restarted": int(sum(i["n_restarts"] > 0 for i in infos)),
        "n_unresolved": int(sum(i["unresolved"] for i in infos)),
        "test_r2_ci95": boot_ci(r2s),
    }
    # baselines (deterministic)
    ybar = ytr.mean(axis=0)
    m, _, alpha = fit_ridge(Xtr, ytr, Xva, yva)
    res["baselines"] = {
        "per_timestep_mean": {"test": eval_all(yte, np.tile(ybar, (len(yte), 1)), 256.0)},
        "ridge": {"alpha": alpha,
                  "test": eval_all(yte, m.predict(Xte.reshape(len(Xte), -1)), 256.0)},
    }
    return res


def main():
    d = load_variant("full")
    X, y, trials, names = stack_all(d)
    results = {"protocol": {"seeds_main": list(SEEDS10), "seeds_cv_abl": list(SEEDS3),
                            "warmup_epochs": WARMUP,
                            "collapse_rule": f"val R2 < {VAL_COLLAPSE_THR} at early stop "
                                             "-> restart seed+1000/+2000, max 3 attempts",
                            "nominal_0.1_note": "mean-predictor collapse sits at val R2 "
                                                "~0.19-0.21, so the operative threshold is 0.3"}}

    print("=== headline split (test=2, val=4, train=1/3/5) ===")
    results["headline_trial2"] = run_split(X, y, trials, names, 2, 4, (1, 3, 5), "headline")
    print("\n=== secondary split (test=5, val=4, train=1/2/3) ===")
    results["secondary_trial5"] = run_split(X, y, trials, names, 5, 4, (1, 2, 3), "secondary")

    # ---------------- CV, seeds 42-44, fold-3 exclusion scoring ----------------
    print("\n=== 5-fold CV (seeds 42-44, restart rule) ===")
    cv_runs, per_fold, cv_infos = [], {}, []
    for f in cv_folds(trials):
        te_names = [names[i] for i in f["test"]]
        excl = te_names.index(EXCLUDE) if EXCLUDE in te_names else None
        keep = [i for i in range(len(te_names)) if i != excl]
        fold_ms = []
        for seed in SEEDS3:
            model, sc, info = train_with_restart(
                build_simple1dcnn, X[f["train"]], y[f["train"]],
                X[f["val"]], y[f["val"]], seed, log_prefix=f"[cv f{f['k']}]")
            p = predict(model, sc, X[f["test"]])
            yte = y[f["test"]]
            mte = eval_all(yte[keep], p[keep], 256.0) if excl is not None \
                else eval_all(yte, p, 256.0)
            fold_ms.append(mte)
            cv_runs.append(mte)
            info["fold"] = f["k"]
            cv_infos.append(info)
            print(f"[cv fold{f['k']} slot {seed}] test R2={mte['r2']:+.3f} "
                  f"restarts={info['n_restarts']}{' UNRESOLVED' if info['unresolved'] else ''}")
        per_fold[f"fold{f['k']}"] = agg(fold_ms)
    r2s = [m["r2"] for m in cv_runs]
    results["cv"] = {
        "test_overall": agg(cv_runs), "per_fold": per_fold,
        "r2_all_runs": [float(m["r2"]) for m in cv_runs],
        "r2_ci95": boot_ci(r2s),
        "slots": cv_infos,
        "n_slots_restarted": int(sum(i["n_restarts"] > 0 for i in cv_infos)),
        "n_unresolved": int(sum(i["unresolved"] for i in cv_infos)),
        "note": "fold 3 scored excluding On_set2_trail3 (audit exclusion)",
    }

    # ---------------- ablation, seeds 42-44 ----------------
    print("\n=== ROI ablation (seeds 42-44, restart rule) ===")
    from run_ablation import reduce_channels
    keep_tr = np.array([i for i, n in enumerate(
        [r["name"] for r in common.reconstruct_row_metadata()["train"]]) if n != EXCLUDE])
    abl = {}
    for mode in ("grid3x3_pol", "grid3x3_nopol", "global_pol", "global_nopol"):
        Xtr = reduce_channels(d["X_train"], mode)[keep_tr]
        ytr = d["y_train"][keep_tr]
        Xva = reduce_channels(d["X_val"], mode)
        Xte = reduce_channels(d["X_test"], mode)
        C = Xtr.shape[1]
        build = partial(build_simple1dcnn, in_channels=C)
        per_test, infos = [], []
        for seed in SEEDS3:
            model, sc, info = train_with_restart(build, Xtr, ytr, Xva, d["y_val"],
                                                 seed, log_prefix=f"[abl {mode}]")
            mte = eval_all(d["y_test"], predict(model, sc, Xte), 256.0)
            per_test.append(mte)
            infos.append(info)
            print(f"[abl {mode:14s} slot {seed}] test R2={mte['r2']:+.3f} "
                  f"restarts={info['n_restarts']}{' UNRESOLVED' if info['unresolved'] else ''}")
        abl[mode] = {"in_channels": int(C), "test": agg(per_test),
                     "slots": infos,
                     "n_slots_restarted": int(sum(i["n_restarts"] > 0 for i in infos)),
                     "n_unresolved": int(sum(i["unresolved"] for i in infos))}
    results["ablation"] = {"note": "fixed trial-5 split (same as original ablation run)",
                           "variants": abl}

    save_json(results, HERE / "seed_protocol_results.json")
    print("\nSaved experiments/seed_protocol_results.json")
    for k in ("headline_trial2", "secondary_trial5"):
        r = results[k]
        print(f"{k}: test R2 {r['test']['r2'][0]:+.3f} ± {r['test']['r2'][1]:.3f} "
              f"CI {np.round(r['test_r2_ci95'],3).tolist()} | restarts {r['n_slots_restarted']}/10 "
              f"unresolved {r['n_unresolved']}")
    r = results["cv"]
    print(f"cv pooled: R2 {r['test_overall']['r2'][0]:+.3f} ± {r['test_overall']['r2'][1]:.3f} "
          f"CI {np.round(r['r2_ci95'],3).tolist()} | restarts {r['n_slots_restarted']}/15 "
          f"unresolved {r['n_unresolved']}")


if __name__ == "__main__":
    main()
