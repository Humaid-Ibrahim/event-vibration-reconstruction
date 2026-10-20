"""
ROI-grid sweep ablation: rebuild 5x5 and 10x10 event-rate representations from
raw DV/LDV data and train simple1dcnn under the unified protocol.

Build: repo BeamDataset per mode (train/val/test), event_rate, polarity split,
align=True, 2048 bins / 8 s — exactly the save_processed_data.py recipe, so the
row ordering matches processed_data.npz (same seed-42 split shuffles).
Correctness check: a 3x3 rebuild of train row 0 must match processed_data.npz
bit-for-bit before the finer grids are trusted.

Train: headline split (test = trial 2, val = trial 4, train = trials {1,3,5}
minus On_set2_trail3), seeds 42-44 with restart-on-collapse (val R2 < 0.3),
warmup 5 -> 5e-4 cosine, ES patience 15, val-only selection.

Outputs: processed_data_grid5.npz, processed_data_grid10.npz (repo root),
         experiments/grid_ablation_results.json
"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
from common import (agg, build_simple1dcnn, eval_all, load_variant,
                    predict, save_json, stack_all)
from run_seed_protocol import train_with_restart, SEEDS3, EXCLUDE  # noqa: E402

GRIDS = {5: ROOT / "processed_data_grid5.npz", 10: ROOT / "processed_data_grid10.npz"}


def build_grid_npz(g, out_path):
    from data_loaders.beam_dataloader_aligned import BeamDataset
    print(f"--- building {out_path.name} (grid {g}x{g}, {2*g*g} ch) ---", flush=True)
    data = {}
    for mode in ("train", "val", "test"):
        ds = BeamDataset(data_root=str(ROOT / "data" / "XJTU-DV-Beam"), mode=mode,
                         event_representation="event_rate", align=True,
                         roi_grid=(g, g), use_polarity_event_rate=True,
                         cache_events=False)
        X, y = [], []
        for i in range(len(ds)):
            xi, yi, _ = ds[i]
            X.append(xi.numpy())
            y.append(yi.numpy())
            print(f"  [{mode} {i+1}/{len(ds)}]", flush=True)
        data[f"X_{mode}"] = np.array(X, dtype=np.float32)
        data[f"y_{mode}"] = np.array(y, dtype=np.float32)
    np.savez_compressed(out_path, **data,
                        config=f"grid{g}x{g}_polarity_2048bins_align",
                        model_name="simple1dcnn_grid_ablation")
    print(f"saved {out_path.name}: X_train {data['X_train'].shape}")
    return data


def verify_3x3_replication():
    from data_loaders.beam_dataloader_aligned import BeamDataset
    ref = np.load(ROOT / "processed_data.npz")
    ds = BeamDataset(data_root=str(ROOT / "data" / "XJTU-DV-Beam"), mode="train",
                     event_representation="event_rate", align=True,
                     roi_grid=(3, 3), use_polarity_event_rate=True,
                     cache_events=False)
    x0, y0, _ = ds[0]
    dx = float(np.abs(x0.numpy() - ref["X_train"][0]).max())
    dy = float(np.abs(y0.numpy() - ref["y_train"][0]).max())
    print(f"3x3 replication check (train row 0): max|dX|={dx:.2e}, max|dy|={dy:.2e}")
    assert dx == 0.0 and dy == 0.0, "pipeline replication FAILED - do not trust finer grids"


def headline_masks(trials, names):
    test_m = trials == 2
    val_m = trials == 4
    train_m = np.array([(t in (1, 3, 5)) and n != EXCLUDE
                        for t, n in zip(trials, names)])
    return train_m, val_m, test_m


def train_grid(X, y, trials, names, label, in_ch):
    train_m, val_m, test_m = headline_masks(trials, names)
    Xtr, ytr = X[train_m], y[train_m]
    Xva, yva = X[val_m], y[val_m]
    Xte, yte = X[test_m], y[test_m]
    build = partial(build_simple1dcnn, in_channels=in_ch)
    n_par = common.n_params(build())
    per, infos = [], []
    for seed in SEEDS3:
        model, sc, info = train_with_restart(build, Xtr, ytr, Xva, yva, seed,
                                             log_prefix=f"[{label}]")
        mte = eval_all(yte, predict(model, sc, Xte), 256.0)
        per.append(mte)
        infos.append(info)
        print(f"[{label} slot {seed}] val={info['final_val_r2']:+.3f} "
              f"test R2={mte['r2']:+.3f} restarts={info['n_restarts']}"
              f"{' UNRESOLVED' if info['unresolved'] else ''}", flush=True)
    return {"n_params": n_par, "in_channels": in_ch, "test": agg(per),
            "test_per_slot": per, "slots": infos,
            "n_slots_restarted": int(sum(i["n_restarts"] > 0 for i in infos)),
            "n_unresolved": int(sum(i["unresolved"] for i in infos))}


def main():
    verify_3x3_replication()

    # metadata (row order identical across all grid files by construction)
    md = common.reconstruct_row_metadata()
    rows = md["train"] + md["val"] + md["test"]
    trials = np.array([r["trial"] for r in rows])
    names = [r["name"] for r in rows]

    results = {"protocol": "headline split, seeds 42-44, restart-on-collapse "
                           "(val R2 < 0.3), warmup+cosine, ES 15", "grids": {}}

    for g, path in GRIDS.items():
        if path.exists():
            d = dict(np.load(path))
            print(f"{path.name} exists - reusing")
        else:
            d = build_grid_npz(g, path)
        X = np.concatenate([d["X_train"], d["X_val"], d["X_test"]]).astype(np.float32)
        y = np.concatenate([d["y_train"], d["y_val"], d["y_test"]]).astype(np.float32)
        # y must match the 3x3 pipeline's targets exactly (same LDV resampling)
        ref = load_variant("full")
        yref = np.concatenate([ref["y_train"], ref["y_val"], ref["y_test"]])
        assert np.allclose(y, yref, atol=1e-6), f"y mismatch for grid {g}"
        results["grids"][f"grid{g}x{g}"] = train_grid(X, y, trials, names,
                                                      f"g{g}", 2 * g * g)
        save_json(results, HERE / "grid_ablation_results.json")

    # ---- split-consistent rows: 1x1 and 3x3 variants ON THE HEADLINE SPLIT.
    # (The previous ablation numbers 0.747/0.644/-0.03 are from the trial-5
    # split; the sweep must not mix splits.)
    from run_ablation import reduce_channels
    ref = load_variant("full")
    X3 = np.concatenate([ref["X_train"], ref["X_val"], ref["X_test"]])
    y3 = np.concatenate([ref["y_train"], ref["y_val"], ref["y_test"]])
    results["grids"]["grid3x3_nopol"] = train_grid(
        reduce_channels(X3, "grid3x3_nopol"), y3, trials, names, "3x3nopol", 9)
    results["grids"]["grid1x1"] = train_grid(
        reduce_channels(X3, "global_pol"), y3, trials, names, "1x1", 2)

    # 3x3 + polarity on the headline split = seeds 42-44 subset of the
    # 10-slot seed-protocol headline run (identical protocol and data)
    sp = json.load(open(HERE / "seed_protocol_results.json"))
    h = sp["headline_trial2"]
    sub = h["test_per_slot"][:3]
    assert [s["slot_seed"] for s in h["slots"][:3]] == [42, 43, 44]
    results["grids"]["grid3x3_pol"] = {
        "n_params": 34872, "in_channels": 18, "test": agg(sub),
        "test_per_slot": sub, "source": "seed_protocol headline slots 42-44",
        "n_slots_restarted": int(sum(s["n_restarts"] > 0 for s in h["slots"][:3])),
        "n_unresolved": 0}
    results["context_trial5_split"] = {
        k: {"test_r2": v["test"]["r2"], "in_channels": v["in_channels"]}
        for k, v in sp["ablation"]["variants"].items()}
    save_json(results, HERE / "grid_ablation_results.json")

    print("\n=== GRID SWEEP (headline split, 3 slots) ===")
    for key in ("grid1x1", "grid3x3_nopol", "grid3x3_pol", "grid5x5", "grid10x10"):
        r = results["grids"][key]
        t = r["test"]
        print(f"{key:14s} ({r['in_channels']:3d} ch, {r['n_params']:>7,} p): "
              f"R2 {t['r2'][0]:+.3f}±{t['r2'][1]:.3f}  spec {t['spec_nrmse'][0]:.3f}  "
              f"decay {t['decay_err'][0]:.3f}  env {t['env_r2'][0]:+.3f}")


if __name__ == "__main__":
    main()
