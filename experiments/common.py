"""
Shared utilities for the P0 experiments (SAFEPROCESS 2027 plan).

- Data loading for the four model variants (full 2048 / 128 / 32-square outputs)
- Reconstruction of the per-row (condition, trial) mapping so 5-fold CV can
  split by trial (replicates _split_data + np.random.seed(42) shuffles from
  data_loaders/beam_dataloader_aligned.py:837-908)
- Train-only StandardScaler fitting (pattern of utils/trainer.py:114-150)
- Compact training loop: AdamW lr 5e-4 wd 1e-3, MSE on scaled y, early
  stopping patience 15 on val loss, max 150 epochs, model selection on val
- Metrics: time-domain R2, normalized spectral RMSE (0-40 Hz capped at
  Nyquist), decay-rate error (slope of log Hilbert envelope), envelope R2,
  2x f0 doubling-artifact ratio (only where 2*f0 < Nyquist)
- Model builders for: simple1dcnn (35K), cnn2d_full (3.13M), speck18k, speck3k5
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_OUTLIER_ROW = 16  # audit 2: train sample 16 misaligned by -352 ms

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_FILES = {
    "full": "processed_data.npz",        # X (N,18,2048), y (N,2048)
    "y128": "processed_data_128.npz",    # X (N,18,2048), y (N,128)
    "sq32": "processed_data_square.npz", # X (N,2,32,32), y (N,32)
}


def load_variant(variant):
    d = np.load(ROOT / DATA_FILES[variant], allow_pickle=True)
    out = {}
    for s in ("train", "val", "test"):
        out[f"X_{s}"] = d[f"X_{s}"].astype(np.float32)
        out[f"y_{s}"] = d[f"y_{s}"].astype(np.float32)
    out["T_out"] = out["y_train"].shape[1]
    out["fs"] = out["T_out"] / 8.0
    return out


# ---------------------------------------------------------------------------
# (condition, trial) mapping — replicates the committed dataloader split
# ---------------------------------------------------------------------------

def _build_data_pairs_names():
    """Order of data_pairs in beam_dataloader_aligned._get_data_pairs():
    sorted stems with matching DV/LDV names, then the typo pair appended last.
    DV stem 'On_set1_trai4' has no LDV match, LDV 'On_set1_trail4' no DV match;
    the special case appends the pair named 'On_set1_trail4' at the end."""
    names = []
    for illum in ("Off", "On"):
        for st in ("set1", "set2", "set3"):
            for tr in range(1, 6):
                if illum == "On" and st == "set1" and tr == 4:
                    continue  # typo pair, appended last
                names.append(f"{illum}_{st}_trail{tr}")
    names.append("On_set1_trail4")
    return names


def reconstruct_row_metadata():
    """Returns dict with, per split, a list of (name, condition, trial) in the
    exact row order of the saved npz files."""
    names = _build_data_pairs_names()

    config_groups = {}
    for idx, name in enumerate(names):
        illum, st, tr = name.split("_")
        trial = int(tr.replace("trail", ""))
        cfg = f"{illum}_{st}"
        config_groups.setdefault(cfg, {1: [], 2: [], 3: [], 4: [], 5: []})
        config_groups[cfg][trial].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for cfg, trials in config_groups.items():
        for t in (1, 2, 3):
            train_idx.extend(trials.get(t, []))
        val_idx.extend(trials.get(4, []))
        test_idx.extend(trials.get(5, []))

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # identical RNG sequence to _split_data (lines 904-908)
    np.random.seed(42)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    def meta(indices):
        out = []
        for i in indices:
            n = names[i]
            illum, st, tr = n.split("_")
            out.append({"name": n, "condition": f"{illum}_{st}",
                        "trial": int(tr.replace("trail", ""))})
        return out

    return {"train": meta(train_idx), "val": meta(val_idx), "test": meta(test_idx)}


def stack_all(data):
    """Concatenate splits -> X_all, y_all plus per-row trial/condition labels."""
    md = reconstruct_row_metadata()
    X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
    y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
    rows = md["train"] + md["val"] + md["test"]
    trials = np.array([r["trial"] for r in rows])
    names = [r["name"] for r in rows]
    return X, y, trials, names


def cv_folds(trials):
    """fold k (k=1..5): test = trial k, val = trial k%5+1, train = the rest."""
    folds = []
    for k in range(1, 6):
        v = k % 5 + 1
        test_m = trials == k
        val_m = trials == v
        train_m = ~(test_m | val_m)
        folds.append({"k": k, "train": np.where(train_m)[0],
                      "val": np.where(val_m)[0], "test": np.where(test_m)[0]})
    return folds


# ---------------------------------------------------------------------------
# Scaling (train-only fit, mirrors utils/trainer.py:114-150)
# ---------------------------------------------------------------------------

class Scalers:
    def __init__(self, X_train, y_train):
        self.xs = StandardScaler().fit(X_train.reshape(len(X_train), -1))
        self.ys = StandardScaler().fit(y_train)

    def x(self, X):
        return self.xs.transform(X.reshape(len(X), -1)).reshape(X.shape).astype(np.float32)

    def y(self, y):
        return self.ys.transform(y).astype(np.float32)

    def y_inv(self, y):
        return self.ys.inverse_transform(y)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_simple1dcnn(dropout=0.5, in_channels=18):
    import sys
    sys.path.insert(0, str(ROOT))
    from models.cnn1d import Simple1DCNN
    return Simple1DCNN(in_channels=in_channels, output_size=2048, dropout=dropout)


class Reshape2D(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.h, self.w = h, w

    def forward(self, x):  # (B, C, T) -> (B, C, h, w)
        return x.view(x.size(0), x.size(1), self.h, self.w)


def build_cnn2d_full():
    """3,133,864-param model matching results/.../best_model.pth layer shapes."""
    return nn.Sequential(
        Reshape2D(32, 64),
        nn.Conv2d(18, 20, 5, 1, 2, bias=False), nn.ReLU(), nn.AvgPool2d(2, 2),
        nn.Conv2d(20, 32, 5, 1, 2, bias=False), nn.ReLU(), nn.AvgPool2d(2, 2),
        nn.Conv2d(32, 128, 3, 1, 1, bias=False), nn.ReLU(), nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(4096, 500, bias=False), nn.ReLU(),
        nn.Linear(500, 2048, bias=False),
    )


def build_speck18k():
    """SPECK_DEPLOYMENT.md architecture: 18,832 params, output 128."""
    return nn.Sequential(
        Reshape2D(32, 64),
        nn.Conv2d(18, 8, 3, 1, 1, bias=False), nn.ReLU(), nn.AvgPool2d(4, 4),
        nn.Conv2d(8, 16, 3, 1, 1, bias=False), nn.ReLU(), nn.AvgPool2d(4, 4),
        nn.Flatten(),
        nn.Linear(128, 128, bias=False),
    )


def build_speck3k5():
    """models/cnn2d_snn_eventrate.py square variant: 3,344 params, output 32.
    Input is already (B, 2, 32, 32) in processed_data_square.npz."""
    return nn.Sequential(
        nn.Conv2d(2, 8, 3, 1, 1, bias=False), nn.ReLU(), nn.AvgPool2d(4, 4),
        nn.Conv2d(8, 16, 3, 1, 1, bias=False), nn.ReLU(), nn.AvgPool2d(4, 4),
        nn.Flatten(),
        nn.Linear(64, 32, bias=False),
    )


MODEL_BUILDERS = {
    "simple1dcnn": (build_simple1dcnn, "full"),
    "cnn2d_full": (build_cnn2d_full, "full"),
    "speck18k": (build_speck18k, "y128"),
    "speck3k5": (build_speck3k5, "sq32"),
}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(build_fn, Xtr, ytr, Xva, yva, seed, epochs=150, patience=15,
                lr=5e-4, weight_decay=1e-3, batch_size=16, warmup_epochs=0):
    """Returns (model, scalers, best_val_loss, epochs_run). Model selection on
    val loss (scaled MSE) only; test is never touched here.

    warmup_epochs > 0 enables linear LR warmup (0 -> lr over warmup_epochs)
    followed by cosine decay to 0 over the remaining epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    sc = Scalers(Xtr, ytr)
    Xtr_s = torch.from_numpy(sc.x(Xtr))
    ytr_s = torch.from_numpy(sc.y(ytr))
    Xva_s = torch.from_numpy(sc.x(Xva)).to(DEVICE)
    yva_s = torch.from_numpy(sc.y(yva)).to(DEVICE)

    model = build_fn().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if warmup_epochs > 0:
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            prog = (ep - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossf = nn.MSELoss()

    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    n = len(Xtr_s)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr_s[idx].to(DEVICE)
            yb = ytr_s[idx].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva_s), yva_s).item()
        if vl < best_val - 1e-6:
            best_val, best_epoch, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, sc, best_val, best_epoch + 1


def predict(model, sc, X, batch_size=32):
    model.eval()
    outs = []
    with torch.no_grad():
        Xs = torch.from_numpy(sc.x(X))
        for i in range(0, len(Xs), batch_size):
            outs.append(model(Xs[i:i + batch_size].to(DEVICE)).cpu().numpy())
    return sc.y_inv(np.concatenate(outs, axis=0))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

RIDGE_ALPHAS = [10.0 ** k for k in range(-5, 7)]


def fit_ridge(Xtr, ytr, Xva, yva):
    Ftr, Fva = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)
    best = (None, -np.inf, None)
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a, solver="svd")
        m.fit(Ftr, ytr)
        r2 = r2_score(yva.ravel(), m.predict(Fva).ravel())
        if r2 > best[1]:
            best = (m, r2, a)
    return best  # (model, val_r2, alpha)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smooth_env(y, fs):
    """Smoothed Hilbert envelope; ~100 ms Gaussian (min sigma 1 bin)."""
    env = np.abs(hilbert(y, axis=-1))
    sigma = max(0.1 * fs, 1.0)
    return gaussian_filter1d(env, sigma, axis=-1)


def decay_slope(y, fs):
    """Slope (1/s) of linear fit to log smoothed Hilbert envelope, per row."""
    env = smooth_env(np.atleast_2d(y), fs)
    t = np.arange(env.shape[1]) / fs
    logs = np.log(np.maximum(env, 1e-8))
    slopes = []
    for row in logs:
        A = np.vstack([t, np.ones_like(t)]).T
        slope, _ = np.linalg.lstsq(A, row, rcond=None)[0]
        slopes.append(slope)
    return np.array(slopes)


def spectral_nrmse(y_true, y_pred, fs, f_hi=40.0):
    """Per-row ||S_pred - S_true|| / ||S_true|| over 0..min(f_hi, Nyquist)."""
    n = y_true.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = freqs <= min(f_hi, fs / 2)
    St = np.abs(np.fft.rfft(y_true, axis=1))[:, band]
    Sp = np.abs(np.fft.rfft(y_pred, axis=1))[:, band]
    return np.linalg.norm(Sp - St, axis=1) / (np.linalg.norm(St, axis=1) + 1e-12)


def doubling_ratio(y, fs, f0_band=(5.0, 9.0), half_bw=0.5):
    """Per-row ratio P(2*f0)/P(f0) of spectral magnitude; f0 located per row
    from the signal itself within f0_band. Requires 2*f0 < Nyquist."""
    n = y.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    S = np.abs(np.fft.rfft(y - y.mean(axis=1, keepdims=True), axis=1))
    ratios, f0s = [], []
    sel = (freqs >= f0_band[0]) & (freqs <= f0_band[1])
    for row in S:
        f0 = freqs[sel][np.argmax(row[sel])]
        if 2 * f0 >= fs / 2:
            ratios.append(np.nan)
            f0s.append(f0)
            continue
        b1 = (freqs >= f0 - half_bw) & (freqs <= f0 + half_bw)
        b2 = (freqs >= 2 * f0 - half_bw) & (freqs <= 2 * f0 + half_bw)
        ratios.append(float(row[b2].max() / (row[b1].max() + 1e-12)))
        f0s.append(float(f0))
    return np.array(ratios), np.array(f0s)


def eval_all(y_true, y_pred, fs):
    """Full metric set for one split. Returns plain-float dict."""
    out = {
        "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
        "mse": float(mean_squared_error(y_true.ravel(), y_pred.ravel())),
        "mae": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "spec_nrmse": float(spectral_nrmse(y_true, y_pred, fs).mean()),
    }
    st, sp = decay_slope(y_true, fs), decay_slope(y_pred, fs)
    out["decay_err"] = float(np.abs(sp - st).mean())
    et, ep = smooth_env(y_true, fs), smooth_env(y_pred, fs)
    out["env_r2"] = float(r2_score(et.ravel(), ep.ravel()))
    if fs / 2 > 14.0:  # 2*f0 (~13.6 Hz) representable
        rt, _ = doubling_ratio(y_true, fs)
        rp, f0s = doubling_ratio(y_pred, fs)
        out["dbl_ratio_true"] = float(np.nanmean(rt))
        out["dbl_ratio_pred"] = float(np.nanmean(rp))
    return out


def agg(dicts):
    """mean ± std across a list of metric dicts -> {key: [mean, std]}."""
    keys = dicts[0].keys()
    return {k: [float(np.mean([d[k] for d in dicts])),
                float(np.std([d[k] for d in dicts]))] for k in keys}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
