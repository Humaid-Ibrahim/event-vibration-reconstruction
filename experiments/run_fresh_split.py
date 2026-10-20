"""
Fresh-test-split experiment (burn-in elimination).

Split: test = trial 2 (never used in any earlier development decision),
       val = trial 4, train = trials {1, 3, 5} minus the excluded
       On_set2_trail3 (17 recordings).
Protocol: identical to run_stability.py's warmup variant — 5-epoch linear
warmup to 5e-4 + cosine, ES patience 15, val-only selection, seeds 0-7.

Outputs: experiments/fresh_split_results.json
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (agg, build_simple1dcnn, eval_all, fit_ridge, load_variant,
                    predict, save_json, stack_all, train_model)

SEEDS = tuple(range(8))
WARMUP = 5
EXCLUDE = "On_set2_trail3"
rng = np.random.default_rng(0)


def boot_ci(x, n_boot=10_000):
    x = np.asarray(x, dtype=float)
    bs = rng.choice(x, size=(n_boot, len(x))).mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    d = load_variant("full")
    X, y, trials, names = stack_all(d)
    fs = d["fs"]

    test_m = trials == 2
    val_m = trials == 4
    train_m = np.array([(t in (1, 3, 5)) and n != EXCLUDE
                        for t, n in zip(trials, names)])
    assert test_m.sum() == 6 and val_m.sum() == 6 and train_m.sum() == 17
    assert not (train_m & (test_m | val_m)).any()
    Xtr, ytr = X[train_m], y[train_m]
    Xva, yva = X[val_m], y[val_m]
    Xte, yte = X[test_m], y[test_m]
    print("train:", sorted(np.array(names)[train_m]))
    print("test :", sorted(np.array(names)[test_m]))

    per_test, per_val, infos = [], [], []
    for seed in SEEDS:
        model, sc, _, ep = train_model(build_simple1dcnn, Xtr, ytr, Xva, yva,
                                       seed, warmup_epochs=WARMUP)
        mva = eval_all(yva, predict(model, sc, Xva), fs)
        mte = eval_all(yte, predict(model, sc, Xte), fs)
        per_val.append(mva)
        per_test.append(mte)
        infos.append({"seed": seed, "epochs_run": ep,
                      "collapsed": bool(mte["r2"] < 0.1)})
        print(f"[fresh s{seed}] ep={ep:3d} val R2={mva['r2']:+.3f} "
              f"test R2={mte['r2']:+.3f} specNRMSE={mte['spec_nrmse']:.3f} "
              f"{'COLLAPSED' if mte['r2'] < 0.1 else ''}")

    r2s = [m["r2"] for m in per_test]
    stable = [r for r in r2s if r >= 0.1]
    results = {
        "split": {"test": "trial 2", "val": "trial 4",
                  "train": "trials 1,3,5 minus On_set2_trail3 (17)"},
        "protocol": {"seeds": list(SEEDS), "warmup_epochs": WARMUP},
        "seeds": infos,
        "val": agg(per_val), "test": agg(per_test),
        "test_per_seed": per_test, "val_per_seed": per_val,
        "n_collapsed": int(sum(i["collapsed"] for i in infos)),
        "test_r2_ci95": boot_ci(r2s),
        "stable_band": {"n": len(stable),
                        "mean": float(np.mean(stable)) if stable else None,
                        "std": float(np.std(stable)) if stable else None,
                        "ci95": boot_ci(stable) if len(stable) >= 2 else None},
    }

    # baselines on the same split
    ybar = ytr.mean(axis=0)
    results["baselines"] = {
        "per_timestep_mean": {
            "val": eval_all(yva, np.tile(ybar, (6, 1)), fs),
            "test": eval_all(yte, np.tile(ybar, (6, 1)), fs)},
    }
    m, _, alpha = fit_ridge(Xtr, ytr, Xva, yva)
    results["baselines"]["ridge"] = {
        "alpha": alpha,
        "val": eval_all(yva, m.predict(Xva.reshape(6, -1)), fs),
        "test": eval_all(yte, m.predict(Xte.reshape(6, -1)), fs)}

    save_json(results, HERE / "fresh_split_results.json")
    print(f"\nfresh split test R2: {np.mean(r2s):+.3f} ± {np.std(r2s):.3f} "
          f"CI {results['test_r2_ci95']}, collapsed {results['n_collapsed']}/8")
    if stable:
        print(f"stable band ({len(stable)} seeds): {np.mean(stable):+.3f} ± {np.std(stable):.3f} "
              f"CI {results['stable_band']['ci95']}")
    for bn in ("ridge", "per_timestep_mean"):
        b = results["baselines"][bn]
        print(f"{bn}: val R2 {b['val']['r2']:+.3f}, test R2 {b['test']['r2']:+.3f}")


if __name__ == "__main__":
    main()
