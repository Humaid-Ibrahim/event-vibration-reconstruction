"""
Expanded architecture comparison on the headline split, single seed protocol.

Protocol (identical to run_seed_protocol.py): test = trial 2, val = trial 4,
train = trials {1,3,5} minus On_set2_trail3 (17); warmup 5 -> 5e-4 cosine,
ES patience 15, val-only selection; 10 slots (seeds 42-51), restart-on-collapse
(val R2 < 0.3, +1000/+2000, max 3 attempts).

Minimal tuning: each new family has <= 2 variants (dropout or lr); the variant
is chosen on VAL ONLY with 3 slots (42-44), then the winner completes all 10
slots. A family whose 3 selection slots are all unresolved for every variant
is stopped and reported as failed.

Families:
  cnn2d_3m       repo 3.13M 2D CNN (lr 5e-4 protocol vs 1e-4 variant)
  wide1dcnn      ~262K stride-1 1D CNN, 128/128/64/32 channels (dropout .5/.2)
  cnn_lstm1d     ~215K 1D conv encoder + 2-layer biLSTM(64) (dropout .5/.2)
  tcn            ~205K dilated residual 1D CNN, dil 1..64, width 64 (dropout .2/.5)
  transformer1d  ~170K conv patch-embed (stride 8) + 2-layer encoder (dropout .1)

simple1dcnn (35K) numbers are reused from seed_protocol_results.json.

Output: experiments/arch_comparison_results.json
"""
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (agg, eval_all, load_variant, predict, save_json,
                    stack_all, train_model)
from run_seed_protocol import train_with_restart, boot_ci, SEEDS10, SEEDS3, EXCLUDE  # noqa: E402

FS = 256.0


# ---------------------------------------------------------------- architectures
def conv_bn(cin, cout, k=9, dropout=0.0):
    layers = [nn.Conv1d(cin, cout, k, 1, k // 2, bias=False),
              nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def build_wide1dcnn(dropout=0.5):
    return nn.Sequential(
        *conv_bn(18, 128, dropout=dropout), *conv_bn(128, 128, dropout=dropout),
        *conv_bn(128, 64, dropout=dropout), *conv_bn(64, 32),
        nn.Conv1d(32, 1, 9, 1, 4, bias=False), nn.Flatten(),
    )


class CNNLSTM1D(nn.Module):
    """1D conv encoder + biLSTM over the 2048-bin sequence + per-step head."""

    def __init__(self, dropout=0.5, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(*conv_bn(18, 64, dropout=dropout),
                                 *conv_bn(64, 64, dropout=dropout))
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):                   # (B, 18, 2048)
        h = self.enc(x).transpose(1, 2)     # (B, 2048, 64)
        h, _ = self.lstm(h)                 # (B, 2048, 128)
        return self.head(h).squeeze(-1)     # (B, 2048)


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation, k=3, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, 1, pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, k, 1, pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


def build_tcn(dropout=0.2, width=64):
    blocks = [TCNBlock(width, d, dropout=dropout) for d in (1, 2, 4, 8, 16, 32, 64)]
    return nn.Sequential(
        nn.Conv1d(18, width, 9, 1, 4, bias=False), nn.BatchNorm1d(width),
        nn.ReLU(inplace=True), *blocks,
        nn.Conv1d(width, 1, 9, 1, 4, bias=False), nn.Flatten(),
    )


class Transformer1D(nn.Module):
    """Conv patch embed (stride 8 -> 256 tokens) + encoder + per-token head."""

    def __init__(self, d=96, nhead=4, layers=2, dropout=0.1, patch=8):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv1d(18, d, kernel_size=9, stride=patch, padding=4)
        self.pos = nn.Parameter(torch.zeros(1, 2048 // patch, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 2 * d, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, patch)

    def forward(self, x):                        # (B, 18, 2048)
        h = self.embed(x).transpose(1, 2)        # (B, 256, d)
        h = self.encoder(h + self.pos)
        out = self.head(h)                       # (B, 256, patch)
        return out.reshape(x.size(0), -1)        # (B, 2048)


FAMILIES = {
    "cnn2d_3m": {"variants": {"lr5e-4": (common.build_cnn2d_full, {}),
                              "lr1e-4": (common.build_cnn2d_full, {"lr": 1e-4})}},
    "wide1dcnn": {"variants": {"do0.5": (partial(build_wide1dcnn, 0.5), {}),
                               "do0.2": (partial(build_wide1dcnn, 0.2), {})}},
    "cnn_lstm1d": {"variants": {"do0.5": (partial(CNNLSTM1D, 0.5), {}),
                                "do0.2": (partial(CNNLSTM1D, 0.2), {})}},
    "tcn": {"variants": {"do0.2": (partial(build_tcn, 0.2), {}),
                         "do0.5": (partial(build_tcn, 0.5), {})}},
    "transformer1d": {"variants": {"do0.1": (Transformer1D, {})}},
}


def run_slots(build_fn, Xtr, ytr, Xva, yva, Xte, yte, seeds, label, train_kw):
    per_test, infos = [], []
    for seed in seeds:
        t0 = time.time()
        model, sc, info = train_with_restart(
            partial_train(build_fn, train_kw), Xtr, ytr, Xva, yva, seed,
            log_prefix=f"[{label}]")
        mte = eval_all(yte, predict(model, sc, Xte), FS)
        per_test.append(mte)
        infos.append(info)
        print(f"[{label} slot {seed}] val={info['final_val_r2']:+.3f} "
              f"test R2={mte['r2']:+.3f} restarts={info['n_restarts']}"
              f"{' UNRESOLVED' if info['unresolved'] else ''} ({time.time()-t0:.0f}s)",
              flush=True)
    return per_test, infos


def partial_train(build_fn, train_kw):
    """train_with_restart calls train_model(build_fn, ...); route extra kwargs
    (e.g. lr) by wrapping train_model via a shim builder tuple."""
    build_fn._train_kw = train_kw  # consumed by the monkeypatched train_model below
    return build_fn


# route per-family lr overrides through train_model
_orig_train_model = train_model


def _train_model_kw(build_fn, *a, **kw):
    extra = getattr(build_fn, "_train_kw", {})
    kw = {**kw, **extra}
    return _orig_train_model(build_fn, *a, **kw)


import run_seed_protocol  # noqa: E402
run_seed_protocol.train_model = _train_model_kw


def main():
    d = load_variant("full")
    X, y, trials, names = stack_all(d)
    test_m = trials == 2
    val_m = trials == 4
    train_m = np.array([(t in (1, 3, 5)) and n != EXCLUDE
                        for t, n in zip(trials, names)])
    Xtr, ytr = X[train_m], y[train_m]
    Xva, yva = X[val_m], y[val_m]
    Xte, yte = X[test_m], y[test_m]

    sp = json.load(open(HERE / "seed_protocol_results.json"))
    results = {"protocol": sp["protocol"] | {"split": "headline trial-2"},
               "families": {}}

    for fam, spec in FAMILIES.items():
        print(f"\n=== {fam} ===", flush=True)
        variants = spec["variants"]
        # --- variant selection on val, 3 slots ---
        sel = {}
        for vname, (bf, tkw) in variants.items():
            n_par = common.n_params(bf())
            per, infos = run_slots(bf, Xtr, ytr, Xva, yva, Xte, yte,
                                   SEEDS3, f"{fam}/{vname}", tkw)
            sel[vname] = {"val_mean": float(np.mean([i["final_val_r2"] for i in infos])),
                          "per": per, "infos": infos, "n_params": n_par,
                          "all_unresolved": all(i["unresolved"] for i in infos)}
            if len(variants) == 1:
                break
        best_v = max(sel, key=lambda v: sel[v]["val_mean"])
        fam_res = {"variants_tried": {v: {"val_mean_3slots": sel[v]["val_mean"],
                                          "all_unresolved": sel[v]["all_unresolved"]}
                                      for v in sel},
                   "chosen_variant": best_v,
                   "n_params": sel[best_v]["n_params"]}

        if all(sel[v]["all_unresolved"] for v in sel):
            # family fails: report the 3-slot numbers of the best-val variant
            fam_res["status"] = "FAILED (all selection slots unresolved for every variant)"
            fam_res["test"] = agg(sel[best_v]["per"])
            fam_res["slots"] = sel[best_v]["infos"]
            fam_res["n_slots"] = 3
            print(f"[{fam}] FAILED at selection stage - stopping at 3 slots")
        else:
            bf, tkw = variants[best_v]
            per, infos = sel[best_v]["per"], sel[best_v]["infos"]
            extra_seeds = [s for s in SEEDS10 if s not in SEEDS3]
            per2, infos2 = run_slots(bf, Xtr, ytr, Xva, yva, Xte, yte,
                                     extra_seeds, f"{fam}/{best_v}", tkw)
            per, infos = per + per2, infos + infos2
            fam_res["status"] = "ok"
            fam_res["test"] = agg(per)
            fam_res["test_per_slot"] = per
            fam_res["slots"] = infos
            fam_res["n_slots"] = len(per)
            fam_res["test_r2_ci95"] = boot_ci([m["r2"] for m in per])
            fam_res["n_slots_restarted"] = int(sum(i["n_restarts"] > 0 for i in infos))
            fam_res["n_unresolved"] = int(sum(i["unresolved"] for i in infos))
        results["families"][fam] = fam_res
        save_json(results, HERE / "arch_comparison_results.json")  # checkpoint

    # reference rows
    results["reference"] = {
        "simple1dcnn": {"n_params": 34872, "test": sp["headline_trial2"]["test"],
                        "test_r2_ci95": sp["headline_trial2"]["test_r2_ci95"],
                        "n_slots_restarted": sp["headline_trial2"]["n_slots_restarted"]},
        "ridge": sp["headline_trial2"]["baselines"]["ridge"]["test"],
        "per_timestep_mean": sp["headline_trial2"]["baselines"]["per_timestep_mean"]["test"],
    }
    save_json(results, HERE / "arch_comparison_results.json")

    print("\n=== SUMMARY (headline split) ===")
    rows = [("simple1dcnn (ref)", 34872, sp["headline_trial2"]["test"], "")]
    for fam, r in results["families"].items():
        rows.append((f"{fam}/{r['chosen_variant']}", r["n_params"], r["test"], r["status"]))
    for name, npar, t, status in rows:
        print(f"{name:24s} {npar:>9,}  R2 {t['r2'][0]:+.3f}±{t['r2'][1]:.3f}  "
              f"spec {t['spec_nrmse'][0]:.3f}  env {t['env_r2'][0]:+.3f}  {status}")


if __name__ == "__main__":
    main()
