"""
Audit 1: Scaler leakage audit.

Question: are the input/output StandardScalers fit on train data only,
or on all data (leakage)?

Evidence collected:
  A. Static code facts (documented in AUDIT_REPORT.md, verified by inspection):
     - utils/trainer.py:114-154  fit_scalers(train_loader) iterates ONLY the
       loader passed in; StandardScaler().fit() on stacked train batches.
     - utils/trainer.py:277-278  train() calls fit_scalers(train_loader, ...)
       with the train loader only.
     - utils/trainer.py:222-223  evaluate() calls _scale_x/_scale_y which use
       .transform() only (no fit) on val/test.
     - apply_scalers.py:34-63    loads a pickled scaler and uses .transform()
       only on all three splits.
  B. Runtime checks (this script):
     - scalers.pkl: n_samples_seen_ for x/y scalers (expect 18 = train size,
       NOT 30 = all data).
     - y_scaler.mean_ compared against y_train.mean(axis=0) (train-only stats)
       vs mean over all 30 samples: which one matches?
     - Per-split mean/std of X and y in processed_data.npz: if val/test were
       suspiciously ~0-mean/1-std, scaling with leaked statistics would be
       indicated.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCALER_PATH = ROOT / "results/cnn2d_snn_eventrate/20260116_081812/scalers.pkl"
DATA_PATH = ROOT / "processed_data.npz"


def main():
    out = {}

    data = np.load(DATA_PATH, allow_pickle=True)
    X = {s: data[f"X_{s}"] for s in ("train", "val", "test")}
    y = {s: data[f"y_{s}"] for s in ("train", "val", "test")}

    # --- B1: per-split statistics of the saved (pre-scaler) data ------------
    print("=" * 70)
    print("Per-split statistics of processed_data.npz (BEFORE trainer scaling)")
    print("=" * 70)
    stats = {}
    for s in ("train", "val", "test"):
        stats[s] = {
            "X_mean": float(X[s].mean()), "X_std": float(X[s].std()),
            "y_mean": float(y[s].mean()), "y_std": float(y[s].std()),
            "n": int(X[s].shape[0]),
        }
        print(f"{s:5s}: X mean={stats[s]['X_mean']:+.5f} std={stats[s]['X_std']:.5f} | "
              f"y mean={stats[s]['y_mean']:+.5f} std={stats[s]['y_std']:.5f} | n={stats[s]['n']}")
    out["split_stats"] = stats

    # Interpretation: X is per-sample abs-max normalized + per-channel mean
    # removed in the dataloader (beam_dataloader_aligned.py:570-582), so X mean
    # ~0 for every split is expected and NOT evidence of leakage.
    # y here should be raw (unscaled) LDV velocity; the scaler is applied later
    # inside the trainer at batch time.

    # --- B2: inspect a trained run's scalers.pkl ----------------------------
    print()
    print("=" * 70)
    print(f"Inspecting {SCALER_PATH.relative_to(ROOT)}")
    print("=" * 70)
    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    xs, ys = scalers["x_scaler"], scalers["y_scaler"]

    sc = {}
    for name, s in (("x_scaler", xs), ("y_scaler", ys)):
        if s is None:
            print(f"{name}: None (skipped)")
            sc[name] = None
            continue
        n_seen = int(np.asarray(s.n_samples_seen_).max())
        sc[name] = {
            "type": type(s).__name__,
            "n_features": int(s.mean_.shape[0]),
            "n_samples_seen": n_seen,
        }
        print(f"{name}: {type(s).__name__}, n_features={s.mean_.shape[0]}, "
              f"n_samples_seen_={n_seen}")
    out["scalers_pkl"] = sc

    n_train, n_all = X["train"].shape[0], sum(v.shape[0] for v in X.values())
    fit_on_train_only = sc["y_scaler"]["n_samples_seen"] == n_train
    print(f"\nn_samples_seen vs split sizes: train={n_train}, all={n_all}")
    print(f"=> scaler fit on {'TRAIN ONLY' if fit_on_train_only else 'MORE THAN TRAIN (check!)'}")
    out["fit_on_train_only_by_count"] = bool(fit_on_train_only)

    # --- B3: does y_scaler.mean_ match train-only stats or all-data stats? --
    if ys is not None and ys.mean_.shape[0] == y["train"].shape[1]:
        y_all = np.concatenate([y["train"], y["val"], y["test"]], axis=0)
        mu_train = y["train"].mean(axis=0)
        mu_all = y_all.mean(axis=0)
        d_train = float(np.abs(ys.mean_ - mu_train).mean())
        d_all = float(np.abs(ys.mean_ - mu_all).mean())
        scale_ref = float(np.abs(y["train"]).mean())
        print(f"\n|y_scaler.mean_ - mean(y_train)|   avg = {d_train:.3e}")
        print(f"|y_scaler.mean_ - mean(y_all 30)|  avg = {d_all:.3e}")
        print(f"(reference magnitude |y_train| avg = {scale_ref:.3e})")
        match = "train" if d_train < d_all else "all"
        print(f"=> y_scaler statistics match the {match.upper()} split")
        out["y_scaler_mean_matches"] = {
            "dist_to_train_mean": d_train, "dist_to_all_mean": d_all,
            "closest": match,
        }
    else:
        out["y_scaler_mean_matches"] = "shape mismatch - different output_size run"
        print("\ny_scaler feature dim != 2048; skipping mean comparison "
              f"(scaler dim = {ys.mean_.shape[0]})")

    # Same comparison for x_scaler if dims match (18*2048 = 36864)
    if xs is not None and xs.mean_.shape[0] == X["train"][0].size:
        X_all = np.concatenate([X[s].reshape(X[s].shape[0], -1) for s in ("train", "val", "test")])
        Xtr = X["train"].reshape(n_train, -1)
        d_train = float(np.abs(xs.mean_ - Xtr.mean(axis=0)).mean())
        d_all = float(np.abs(xs.mean_ - X_all.mean(axis=0)).mean())
        print(f"\n|x_scaler.mean_ - mean(X_train)|  avg = {d_train:.3e}")
        print(f"|x_scaler.mean_ - mean(X_all 30)| avg = {d_all:.3e}")
        match = "train" if d_train < d_all else "all"
        print(f"=> x_scaler statistics match the {match.upper()} split")
        out["x_scaler_mean_matches"] = {
            "dist_to_train_mean": d_train, "dist_to_all_mean": d_all,
            "closest": match,
        }

    # --- verdict -------------------------------------------------------------
    print()
    print("=" * 70)
    verdict = "PASS (no leakage)" if fit_on_train_only else "FAIL (possible leakage)"
    print(f"AUDIT 1 VERDICT: {verdict}")
    print("=" * 70)
    out["verdict"] = verdict

    with open(ROOT / "audits" / "scaler_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved audits/scaler_audit.json")


if __name__ == "__main__":
    main()
