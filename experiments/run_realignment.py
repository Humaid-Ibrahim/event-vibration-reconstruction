"""
Task 3a: realign On_set2_trail3 from the raw DV/LDV files and write
processed_data_realigned.npz (processed_data.npz with train row 16's X
replaced; y and all other rows untouched).

Method:
 1. Instantiate the repo's BeamDataset exactly as save_processed_data.py does
    for the 18-channel config (event_rate, 3x3 ROI grid, polarity, align=True).
 2. Rebuild On_set2_trail3's (18, 2048) representation with the dataset's own
    (faulty) onset-detection offset and verify it matches processed_data.npz
    train row 16 — proving the re-derivation replicates the stored pipeline.
 3. Grid-search a corrected offset (±1.2 s around the faulty one, 1-bin steps)
    maximizing correlation between the smoothed global event rate and the
    smoothed |y| envelope on the 8 s / 2048-bin frame.
 4. Inject the corrected offset into the dataset's offset cache dict, rebuild
    the sample, re-run the audit-2 lag check (should be ~0), and save the
    patched npz.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
from data_loaders.beam_dataloader_aligned import BeamDataset  # noqa: E402

NAME = "On_set2_trail3"
TRAIN_ROW = 16
SIG = 12.8  # ~50 ms envelope smoothing on the 2048-bin grid


def global_rate_for_offset(events_t_sec, offset_s):
    """Global event count histogram on the 8 s / 2048 grid for a given offset."""
    t = events_t_sec - offset_s
    keep = (t >= 0) & (t <= 8.0)
    counts, _ = np.histogram(t[keep], bins=np.linspace(0, 8.0, 2049))
    return counts.astype(np.float32)


def corr_envelope(rate, y_env):
    r = gaussian_filter1d(rate, SIG)
    r = (r - r.mean()) / (r.std() + 1e-12)
    e = (y_env - y_env.mean()) / (y_env.std() + 1e-12)
    return float(np.dot(r, e) / len(r))


def audit_lag(x18, y, max_lag=256):
    """Audit-2 check: best lag between sum-of-channels envelope and |y| env."""
    g = gaussian_filter1d(np.abs(x18.sum(0)), SIG)
    e = gaussian_filter1d(np.abs(y), SIG)
    g = (g - g.mean()) / (g.std() + 1e-12)
    e = (e - e.mean()) / (e.std() + 1e-12)
    n = len(g)
    best = (0, 0.0)
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            a, b = g[: n - L], e[L:]
        else:
            a, b = g[-L:], e[: n + L]
        c = float(np.dot(a, b) / len(a))
        if abs(c) > abs(best[1]):
            best = (L, c)
    return best


def main():
    ds = BeamDataset(
        data_root=str(ROOT / "data" / "XJTU-DV-Beam"),
        mode="all", event_representation="event_rate", align=True,
        roi_grid=(3, 3), use_polarity_event_rate=True, cache_events=False,
    )
    pos = [i for i, gi in enumerate(ds.indices)
           if ds.data_pairs[gi]["name"] == NAME]
    assert len(pos) == 1, f"{NAME} not found"
    idx = pos[0]
    pair = ds.data_pairs[ds.indices[idx]]

    # --- step 2: rebuild with the pipeline's own offset; verify vs stored row
    x_bad, y_t, _ = ds[idx]
    x_bad, y = x_bad.numpy(), y_t.numpy()
    stored = np.load(ROOT / "processed_data.npz")["X_train"][TRAIN_ROW]
    stored_y = np.load(ROOT / "processed_data.npz")["y_train"][TRAIN_ROW]
    dx = float(np.abs(x_bad - stored).max())
    dy = float(np.abs(y - stored_y).max())
    print(f"replication check vs npz train row {TRAIN_ROW}: max|dX|={dx:.2e}, max|dy|={dy:.2e}")
    faulty_offset = float(ds.offsets[NAME]["offset_s"])
    print(f"pipeline (faulty) offset: {faulty_offset:.4f} s")
    lag0, corr0 = audit_lag(x_bad, y)
    print(f"audit lag BEFORE: {lag0} bins ({lag0*8000/2048:+.1f} ms), corr {corr0:+.3f}")

    # --- step 3: corrected offset. Initialize by matching the global-rate
    # peak to the |y|-envelope peak on their raw timelines (robust for an
    # impulse-ringdown), then refine ±0.5 s by envelope correlation computed
    # ONLY over the aligned region actually covered by events (this recording's
    # camera started ~3.4 s after the LDV window; the empty prefix otherwise
    # dominates a naive correlation).
    events = ds._load_events(pair["dv"], None)
    t_sec = ds._to_seconds_events(events["t"])
    y_env = gaussian_filter1d(np.abs(y), SIG)
    bin_s = 8.0 / 2048.0
    centers = (np.arange(2048) + 0.5) * bin_s

    edges10 = np.arange(t_sec.min(), t_sec.max() + 0.01, 0.01)
    rate10 = gaussian_filter1d(np.histogram(t_sec, bins=edges10)[0].astype(float), 5)
    tpk_ev = float(edges10[np.argmax(rate10)])
    tpk_y = float(centers[np.argmax(y_env)])
    init_offset = tpk_ev - tpk_y
    print(f"peak-match init: event peak {tpk_ev:.3f} s raw, |y| peak {tpk_y:.3f} s "
          f"-> init offset {init_offset:.4f} s")

    def score(offset):
        rate = gaussian_filter1d(global_rate_for_offset(t_sec, offset), SIG)
        cover = centers > max(0.0, t_sec.min() - offset) + 0.2
        r, e = rate[cover], y_env[cover]
        r = (r - r.mean()) / (r.std() + 1e-12)
        e = (e - e.mean()) / (e.std() + 1e-12)
        return float(np.dot(r, e) / len(r))

    cands = init_offset + np.arange(-0.5, 0.5 + 1e-9, bin_s)
    scores = [score(o) for o in cands]
    best_offset = float(cands[int(np.argmax(scores))])
    print(f"refined offset: {best_offset:.4f} s (corr {max(scores):+.3f}); "
          f"shift vs faulty: {(best_offset-faulty_offset)*1000:+.1f} ms")

    # --- step 4: rebuild with corrected offset, re-audit, save patched npz
    ds.offsets[NAME] = {"offset_s": best_offset, "score": 1.0}
    x_fix_t, y_fix_t, _ = ds[idx]
    x_fix, y_fix = x_fix_t.numpy(), y_fix_t.numpy()
    assert np.allclose(y_fix, y, atol=1e-6), "y should be offset-independent"
    lag1, corr1 = audit_lag(x_fix, y)
    print(f"audit lag AFTER:  {lag1} bins ({lag1*8000/2048:+.1f} ms), corr {corr1:+.3f}")

    d = dict(np.load(ROOT / "processed_data.npz", allow_pickle=True))
    Xtr = d["X_train"].copy()
    Xtr[TRAIN_ROW] = x_fix
    d["X_train"] = Xtr
    np.savez_compressed(ROOT / "processed_data_realigned.npz", **d)
    print(f"wrote processed_data_realigned.npz (train row {TRAIN_ROW} X replaced)")

    common.save_json({
        "recording": NAME, "train_row": TRAIN_ROW,
        "replication_maxabs_dX": dx, "replication_maxabs_dy": dy,
        "faulty_offset_s": faulty_offset, "corrected_offset_s": best_offset,
        "shift_ms": (best_offset - faulty_offset) * 1000,
        "event_coverage_aligned_s": [float(max(0.0, t_sec.min() - best_offset)),
                                     float(min(8.0, t_sec.max() - best_offset))],
        "audit_lag_before_bins": lag0, "audit_corr_before": corr0,
        "audit_lag_after_bins": lag1, "audit_corr_after": corr1,
    }, HERE / "realignment_info.json")


if __name__ == "__main__":
    main()
