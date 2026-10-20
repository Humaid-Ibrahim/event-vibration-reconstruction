"""
Small-variant architecture comparison (matched to simple1dcnn's 35K scale),
same unified protocol as run_arch_comparison.py. cnn2d is dropped entirely.

Small variants (~10-60K params):
  wide1dcnn_s     stride-1 1D CNN 48/48/24 ch, k=9           (~39K; do .5/.2)
  cnn_lstm1d_s    conv 18->32->32 + 2-layer biLSTM(24)       (~40K; do .5/.2)
  tcn_s           dilated residual, width 32, dil 1..64      (~49K; do .2/.5)
  transformer1d_s patch-8 encoder, d=48, 2 layers            (~58K; do .1)

Large-variant results are read from arch_comparison_results.json (not re-run).

Output: experiments/arch_small_results.json
"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import agg, eval_all, load_variant, save_json, stack_all  # noqa: E402
from run_seed_protocol import SEEDS10, SEEDS3, EXCLUDE  # noqa: E402
from run_arch_comparison import (CNNLSTM1D, TCNBlock, Transformer1D, boot_ci,  # noqa: E402
                                 conv_bn, run_slots)


def build_wide1dcnn_s(dropout=0.5):
    return nn.Sequential(
        *conv_bn(18, 48, dropout=dropout), *conv_bn(48, 48, dropout=dropout),
        *conv_bn(48, 24, dropout=dropout),
        nn.Conv1d(24, 1, 9, 1, 4, bias=False), nn.Flatten(),
    )


class CNNLSTM1D_S(CNNLSTM1D):
    def __init__(self, dropout=0.5, hidden=24):
        super().__init__(dropout=dropout, hidden=hidden)
        self.enc = nn.Sequential(*conv_bn(18, 32, dropout=dropout),
                                 *conv_bn(32, 32, dropout=dropout))
        self.lstm = nn.LSTM(32, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, 1)


def build_tcn_s(dropout=0.2, width=32):
    blocks = [TCNBlock(width, d, dropout=dropout) for d in (1, 2, 4, 8, 16, 32, 64)]
    return nn.Sequential(
        nn.Conv1d(18, width, 9, 1, 4, bias=False), nn.BatchNorm1d(width),
        nn.ReLU(inplace=True), *blocks,
        nn.Conv1d(width, 1, 9, 1, 4, bias=False), nn.Flatten(),
    )


FAMILIES_S = {
    "wide1dcnn_s": {"do0.5": (partial(build_wide1dcnn_s, 0.5), {}),
                    "do0.2": (partial(build_wide1dcnn_s, 0.2), {})},
    "cnn_lstm1d_s": {"do0.5": (partial(CNNLSTM1D_S, 0.5), {}),
                     "do0.2": (partial(CNNLSTM1D_S, 0.2), {})},
    "tcn_s": {"do0.2": (partial(build_tcn_s, 0.2), {}),
              "do0.5": (partial(build_tcn_s, 0.5), {})},
    "transformer1d_s": {"do0.1": (partial(Transformer1D, d=48, nhead=4, layers=2,
                                          dropout=0.1), {})},
}


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

    large = json.load(open(HERE / "arch_comparison_results.json"))
    results = {"protocol": large["protocol"], "families": {}}

    for fam, variants in FAMILIES_S.items():
        print(f"\n=== {fam} ===", flush=True)
        sel = {}
        for vname, (bf, tkw) in variants.items():
            print(f"[{fam}/{vname}] params = {common.n_params(bf()):,}")
            per, infos = run_slots(bf, Xtr, ytr, Xva, yva, Xte, yte,
                                   SEEDS3, f"{fam}/{vname}", tkw)
            sel[vname] = {"val_mean": float(np.mean([i["final_val_r2"] for i in infos])),
                          "per": per, "infos": infos,
                          "n_params": common.n_params(bf()),
                          "all_unresolved": all(i["unresolved"] for i in infos)}
            if len(variants) == 1:
                break
        best_v = max(sel, key=lambda v: sel[v]["val_mean"])
        fam_res = {"variants_tried": {v: {"val_mean_3slots": sel[v]["val_mean"],
                                          "all_unresolved": sel[v]["all_unresolved"]}
                                      for v in sel},
                   "chosen_variant": best_v, "n_params": sel[best_v]["n_params"]}
        if all(sel[v]["all_unresolved"] for v in sel):
            fam_res["status"] = "FAILED (all selection slots unresolved)"
            fam_res["test"] = agg(sel[best_v]["per"])
            fam_res["slots"] = sel[best_v]["infos"]
            fam_res["n_slots"] = 3
        else:
            bf, tkw = variants[best_v]
            per, infos = sel[best_v]["per"], sel[best_v]["infos"]
            extra = [s for s in SEEDS10 if s not in SEEDS3]
            per2, infos2 = run_slots(bf, Xtr, ytr, Xva, yva, Xte, yte,
                                     extra, f"{fam}/{best_v}", tkw)
            per, infos = per + per2, infos + infos2
            fam_res.update({"status": "ok", "test": agg(per), "test_per_slot": per,
                            "slots": infos, "n_slots": len(per),
                            "test_r2_ci95": boot_ci([m["r2"] for m in per]),
                            "n_slots_restarted": int(sum(i["n_restarts"] > 0 for i in infos)),
                            "n_unresolved": int(sum(i["unresolved"] for i in infos))})
        results["families"][fam] = fam_res
        save_json(results, HERE / "arch_small_results.json")

    # ---- combined size-sweep table (small vs large vs reference) ----
    results["large_variants"] = {k: {"n_params": v["n_params"],
                                     "chosen_variant": v["chosen_variant"],
                                     "test_r2": v["test"]["r2"],
                                     "status": v["status"]}
                                 for k, v in large["families"].items()
                                 if k != "cnn2d_3m"}
    results["reference_simple1dcnn"] = large["reference"]["simple1dcnn"]
    results["baselines"] = {k: large["reference"][k]
                            for k in ("ridge", "per_timestep_mean")}
    save_json(results, HERE / "arch_small_results.json")

    print("\n=== SUMMARY (small variants, headline split) ===")
    ref = results["reference_simple1dcnn"]
    print(f"{'simple1dcnn (ref)':22s} {ref['n_params']:>8,}  "
          f"R2 {ref['test']['r2'][0]:+.3f}±{ref['test']['r2'][1]:.3f}")
    for fam, r in results["families"].items():
        t = r["test"]
        lv = results["large_variants"].get(fam.replace("_s", ""), {})
        print(f"{fam+'/'+r['chosen_variant']:22s} {r['n_params']:>8,}  "
              f"R2 {t['r2'][0]:+.3f}±{t['r2'][1]:.3f}  spec {t['spec_nrmse'][0]:.3f}  "
              f"env {t['env_r2'][0]:+.3f}  {r['status']}  "
              f"| large: {lv.get('n_params', 0):,} R2 {lv.get('test_r2', ['?'])[0]}")


if __name__ == "__main__":
    main()
