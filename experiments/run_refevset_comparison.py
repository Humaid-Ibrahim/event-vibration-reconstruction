"""
REF-EvSet head-to-head on OUR recordings (task 2 of the post-download work).

Reconstructs REF-EvSet's training-free best-25-patch signed-rate signal per
recording (replicating competitors/REFEvSet-V1/scripts/beam_calibration.py
lines 38-71 exactly), then scores it with OUR metric implementations
(experiments/common.py) so the numbers drop into our tables:

  frame      : onset-relative 6 s window, resampled to 256 Hz
  their recon: per-recording lag search (±0.3 s) + least-squares amplitude fit
               against the LDV segment (their signal has arbitrary units and
               their method only claims the spectrum, so this is the most
               favorable fair treatment; flagged in the report)
  our model  : simple1dcnn best warmup seed (experiments/predictions_simple1dcnn.npz),
               same 6 s onset window, NO per-recording scale or lag fitting
               (it predicts calibrated velocity directly)

Also summarizes their own metrics (outputs/beam_calibration.json) for the
6 test recordings (trial 5) and all 30, and checks the 2.02x f0 artifact.

Output: experiments/refevset_comparison.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REF = ROOT / "competitors" / "REFEvSet-V1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REF))

import common  # noqa: E402
from evset.data.registry import beam_files  # noqa: E402
from evset.features.patch_rate import load_cache  # noqa: E402

FS_EV = 1000.0
FS_OUT = 256.0
WIN_S = 6.0
TEST_TRIAL = 5


def spec_ref(x, fs, fmax=40.0):
    """Their spectrum definition (hanning, 8x zero-pad, max-normalized, 1-40 Hz)."""
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=8 * len(x)))
    f = np.fft.rfftfreq(8 * len(x), 1 / fs)
    m = (f >= 1.0) & (f <= fmax)
    return f[m], X[m] / X[m].max()


def reconstruct(rec):
    """Replicates beam_calibration.py: returns (sig @1kHz over 6s onset window,
    ldv segment v, ldv times rel. onset, fs_ldv)."""
    z = load_cache(str(REF / "cache" / "beam_l1" / (rec.name + ".npz")))
    c = np.asarray(z["counts"]).astype(np.float32)
    gu = c.sum(axis=(0, 1))
    t_ev = np.arange(len(gu)) / FS_EV
    ldv = np.loadtxt(rec.extra["ldv"], skiprows=5)
    t_l, v = ldv[:, 0], ldv[:, 1]
    fs_l = 1 / np.median(np.diff(t_l))
    on_l = t_l[np.argmax(np.abs(v) > 0.2 * np.abs(v).max())]
    base = np.median(gu[: int(0.5 * FS_EV)]) + 1
    on_e = t_ev[np.argmax(gu > max(5 * base, 0.2 * gu.max()))]
    seg_l = (t_l >= on_l) & (t_l < on_l + WIN_S)
    seg_e = (t_ev >= on_e) & (t_ev < on_e + WIN_S)

    ps = (c[1] - c[0])[:, seg_e]
    f_l, S_l = spec_ref(v[seg_l], fs_l)
    f0 = f_l[np.argmax(S_l)]
    n = ps.shape[1]
    w = np.hanning(n)
    k = np.exp(-2j * np.pi * f0 * np.arange(n) / FS_EV)
    Xf0 = np.abs((ps - ps.mean(1, keepdims=True)) * w @ k) ** 2
    noise = (c[0] + c[1])[:, seg_e].mean(1) * (w ** 2).sum() + 1e-6
    snr = Xf0 / noise
    top = np.argsort(snr)[-25:]
    lead = ps[top[-1]] - ps[top[-1]].mean()
    sig = np.zeros(n)
    for p in top:
        x = ps[p] - ps[p].mean()
        s = np.sign(np.dot(x, lead)) or 1.0
        sig += s * x
    return sig, v[seg_l] - v[seg_l].mean(), (t_l[seg_l] - on_l), fs_l, float(f0)


def to_grid(t_src, x_src, n=int(WIN_S * FS_OUT)):
    tg = np.arange(n) / FS_OUT
    return np.interp(tg, t_src, x_src)


def fit_lag_scale(sig_256, y_256, max_lag=int(0.3 * FS_OUT)):
    """Per-recording lag (±0.3 s) + LS amplitude fit of sig onto y."""
    a = sig_256 - sig_256.mean()
    b = y_256 - y_256.mean()
    best = (0, -np.inf)
    n = len(a)
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            x, z = a[: n - L], b[L:]
        else:
            x, z = a[-L:], b[: n + L]
        c = abs(np.dot(x, z) / (np.linalg.norm(x) * np.linalg.norm(z) + 1e-12))
        if c > best[1]:
            best = (L, c)
    L = best[0]
    if L >= 0:
        x, z = a[: n - L], b[L:]
    else:
        x, z = a[-L:], b[: n + L]
    k = float(np.dot(x, z) / (np.dot(x, x) + 1e-12))
    return k * x, z, L


def onset_window(y, fs=FS_OUT):
    i0 = int(np.argmax(np.abs(y) > 0.2 * np.abs(y).max()))
    n = int(WIN_S * fs)
    i0 = min(i0, len(y) - n)
    return slice(i0, i0 + n)


def our_metrics(y_true, y_pred, fs=FS_OUT):
    m = {
        "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
        "spec_nrmse": float(common.spectral_nrmse(np.atleast_2d(y_true),
                                                  np.atleast_2d(y_pred), fs).mean()),
    }
    st = common.decay_slope(np.atleast_2d(y_true), fs)
    sp = common.decay_slope(np.atleast_2d(y_pred), fs)
    m["decay_err"] = float(np.abs(sp - st).mean())
    et = common.smooth_env(np.atleast_2d(y_true), fs)
    ep = common.smooth_env(np.atleast_2d(y_pred), fs)
    m["env_r2"] = float(r2_score(et.ravel(), ep.ravel()))
    return m


def main():
    recs = {r.name: r for r in beam_files()}
    their = json.load(open(REF / "outputs" / "beam_calibration.json"))["reconstruction"]
    their_by_name = {x["name"]: x for x in their}

    test_names = [f"{il}_set{s}_trail{TEST_TRIAL}" for il in ("Off", "On") for s in (1, 2, 3)]

    # ---- REF-EvSet recon scored with OUR metrics (all 30 + test subset) ----
    ref_rows = []
    for name, rec in sorted(recs.items()):
        sig, v_seg, t_seg, fs_l, f0 = reconstruct(rec)
        sig_256 = to_grid(np.arange(len(sig)) / FS_EV, sig)
        y_256 = to_grid(t_seg, v_seg)
        pred, truth, lag = fit_lag_scale(sig_256, y_256)
        m = our_metrics(truth, pred)
        m["name"] = name
        m["lag_ms"] = lag / FS_OUT * 1000
        ref_rows.append(m)
        print(f"[REF {name:16s}] ourR2={m['r2']:+.3f} specNRMSE={m['spec_nrmse']:.3f} "
              f"decayErr={m['decay_err']:.3f} envR2={m['env_r2']:+.3f} lag={m['lag_ms']:+.0f}ms")

    def norm_name(n):
        return n.replace("trai4", "trail4")

    ref_test = [m for m in ref_rows if norm_name(m["name"]) in test_names]

    # ---- our simple1dcnn on the same 6 s onset windows (no scale/lag fit) ----
    pred_npz = np.load(HERE / "predictions_simple1dcnn.npz")
    yt, yp = pred_npz["y_test"], pred_npz["y_test_pred"]
    md = common.reconstruct_row_metadata()
    row_names = [r["name"] for r in md["test"]]
    ours_rows = []
    for i, name in enumerate(row_names):
        sl = onset_window(yt[i])
        m = our_metrics(yt[i][sl], yp[i][sl])
        m["name"] = name
        ours_rows.append(m)
        print(f"[OURS {name:16s}] ourR2={m['r2']:+.3f} specNRMSE={m['spec_nrmse']:.3f} "
              f"decayErr={m['decay_err']:.3f} envR2={m['env_r2']:+.3f}")

    # ---- their own metrics + doubling check ----
    def their_agg(names=None):
        rows = [x for x in their if names is None or norm_name(x["name"]) in names]
        rm = np.array([x["rmse"] for x in rows])
        fe = np.abs(np.array([x["f0_ldv"] - x["f0_event"] for x in rows]))
        de = np.array([x["decay_event"] - x["decay_ldv"] for x in rows])
        dbl = sum(abs(x["f0_event"] - 2 * x["f0_ldv"]) < 0.5 for x in rows)
        return {"n": len(rows), "spec_rmse_theirdef": [float(rm.mean()), float(rm.std())],
                "f0_abs_err_hz": [float(fe.mean()), float(fe.max())],
                "decay_bias_1_s": [float(de.mean()), float(de.std())],
                "n_doubling_artifact": int(dbl)}

    def agg(rows):
        keys = ("r2", "spec_nrmse", "decay_err", "env_r2")
        return {k: [float(np.mean([r[k] for r in rows])),
                    float(np.std([r[k] for r in rows]))] for k in keys}

    out = {
        "frame": f"onset-relative {WIN_S:.0f}s window @ {FS_OUT:.0f} Hz; REF recon gets "
                 "per-recording lag(±0.3s)+scale fit, ours does not",
        "refevset_ourmetrics_all30": agg(ref_rows),
        "refevset_ourmetrics_test6": agg(ref_test),
        "ours_ourmetrics_test6": agg(ours_rows),
        "refevset_own_metrics_all30": their_agg(),
        "refevset_own_metrics_test6": their_agg(test_names),
        "per_recording": {"refevset": ref_rows, "ours": ours_rows},
    }
    common.save_json(out, HERE / "refevset_comparison.json")

    print("\n=== SUMMARY (test 6 recordings, our metric defs, 6s onset frame) ===")
    for label, a in (("REF-EvSet (lag+scale fitted)", out["refevset_ourmetrics_test6"]),
                     ("simple1dcnn (no fitting)", out["ours_ourmetrics_test6"])):
        print(f"{label:30s} R2 {a['r2'][0]:+.3f}±{a['r2'][1]:.3f} | specNRMSE {a['spec_nrmse'][0]:.3f} | "
              f"decayErr {a['decay_err'][0]:.3f} | envR2 {a['env_r2'][0]:+.3f}")
    t = out["refevset_own_metrics_all30"]
    print(f"\nREF-EvSet own metrics (all 30): specRMSE(their def) {t['spec_rmse_theirdef'][0]:.3f}, "
          f"|f0 err| {t['f0_abs_err_hz'][0]:.3f} Hz, decay bias {t['decay_bias_1_s'][0]:+.2f} 1/s, "
          f"doubling artifacts: {t['n_doubling_artifact']}/30")


if __name__ == "__main__":
    main()
