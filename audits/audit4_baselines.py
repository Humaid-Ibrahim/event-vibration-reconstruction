"""
Audit 4: Trivial baselines on the existing train/val/test split of
processed_data.npz.

Baselines:
  (a) mean predictor        - predict the scalar train mean everywhere
  (b) per-timestep mean     - predict the train-mean signal ybar(t)
  (c) ridge (flattened X)   - 18x2048 = 36864 features -> y(2048); alpha tuned on val
  (d) ridge (channel stats) - per-channel summary stats (mean,std,min,max,energy,
                              p95 of |.|) -> y(2048); alpha tuned on val

Metrics per split: R^2 / MSE / MAE on flattened arrays (matches utils/trainer.py
evaluate()), plus dominant-frequency error (FFT peak in 0.5-40 Hz, fs=256 Hz).

Saves: audits/baseline_metrics.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "processed_data.npz"
FS = 256.0  # 2048 samples / 8 s
FBAND = (0.5, 40.0)
ALPHAS = [10.0 ** k for k in range(-5, 7)]


def dominant_freq(sig_batch):
    """FFT peak frequency in FBAND for each row of (N, T)."""
    n = sig_batch.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    mag = np.abs(np.fft.rfft(sig_batch - sig_batch.mean(axis=1, keepdims=True), axis=1))
    band = (freqs >= FBAND[0]) & (freqs <= FBAND[1])
    fb = freqs[band]
    return fb[np.argmax(mag[:, band], axis=1)]


def metrics(y_true, y_pred):
    out = {
        "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
        "mse": float(mean_squared_error(y_true.ravel(), y_pred.ravel())),
        "mae": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
    }
    f_true = dominant_freq(y_true)
    f_pred = dominant_freq(y_pred)
    out["domfreq_mae_hz"] = float(np.abs(f_true - f_pred).mean())
    out["domfreq_true_hz"] = [float(v) for v in f_true]
    out["domfreq_pred_hz"] = [float(v) for v in f_pred]
    return out


def channel_stats(X):
    """(N, C, T) -> (N, C*6) summary features."""
    feats = [
        X.mean(axis=2), X.std(axis=2), X.min(axis=2), X.max(axis=2),
        (X ** 2).mean(axis=2), np.percentile(np.abs(X), 95, axis=2),
    ]
    return np.concatenate(feats, axis=1)


def tune_ridge(Ftr, ytr, Fva, yva, solver="auto"):
    best = (None, -np.inf, None)
    for a in ALPHAS:
        m = Ridge(alpha=a, solver=solver)
        m.fit(Ftr, ytr)
        r2 = r2_score(yva.ravel(), m.predict(Fva).ravel())
        if r2 > best[1]:
            best = (m, r2, a)
    return best


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]
    print(f"train {Xtr.shape}/{ytr.shape}, val {Xva.shape}, test {Xte.shape}")

    results = {}

    # (a) scalar mean predictor
    mu = float(ytr.mean())
    results["a_mean_predictor"] = {
        "train_mean": mu,
        "val": metrics(yva, np.full_like(yva, mu)),
        "test": metrics(yte, np.full_like(yte, mu)),
    }

    # (b) per-timestep train-mean signal
    ybar = ytr.mean(axis=0)
    results["b_per_timestep_mean"] = {
        "val": metrics(yva, np.tile(ybar, (len(yva), 1))),
        "test": metrics(yte, np.tile(ybar, (len(yte), 1))),
    }

    # (c) ridge on flattened X
    Ftr = Xtr.reshape(len(Xtr), -1)
    Fva = Xva.reshape(len(Xva), -1)
    Fte = Xte.reshape(len(Xte), -1)
    m, r2v, alpha = tune_ridge(Ftr, ytr, Fva, yva, solver="svd")
    results["c_ridge_flattened"] = {
        "alpha": alpha, "n_features": int(Ftr.shape[1]),
        "val": metrics(yva, m.predict(Fva)),
        "test": metrics(yte, m.predict(Fte)),
    }

    # (d) ridge on per-channel summary stats
    Str, Sva, Ste = channel_stats(Xtr), channel_stats(Xva), channel_stats(Xte)
    m, r2v, alpha = tune_ridge(Str, ytr, Sva, yva)
    results["d_ridge_channel_stats"] = {
        "alpha": alpha, "n_features": int(Str.shape[1]),
        "val": metrics(yva, m.predict(Sva)),
        "test": metrics(yte, m.predict(Ste)),
    }

    # ---- print table ----------------------------------------------------------
    print("\n{:<24s} {:>5s} {:>9s} {:>12s} {:>10s} {:>12s}".format(
        "baseline", "split", "R2", "MSE", "MAE", "domF MAE Hz"))
    for name, res in results.items():
        for split in ("val", "test"):
            r = res[split]
            print("{:<24s} {:>5s} {:>+9.4f} {:>12.4e} {:>10.4e} {:>12.2f}".format(
                name, split, r["r2"], r["mse"], r["mae"], r["domfreq_mae_hz"]))

    with open(ROOT / "audits" / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved audits/baseline_metrics.json")


if __name__ == "__main__":
    main()
