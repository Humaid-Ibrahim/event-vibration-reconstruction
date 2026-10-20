"""
P0 item 5: ANN -> SNN conversion with sinabs for the Speck-minimal models.

Procedure per model (speck3k5, speck18k):
 1. Retrain the best fixed-split seed (same recipe as run_fixed_split.py:
    train = trials 1-3 minus outlier row 16, selection on val) — training is
    deterministic given the seed, so this reproduces the reported ANN.
 2. Convert with sinabs.from_torch.from_model (ReLU -> IAF). The final
    bias-free Linear stays linear (regression head read as membrane current).
 3. Rate encoding: the standardized event-rate frame x is presented
    unchanged at every one of N timesteps (constant-current injection);
    IAF neurons then fire at rate ~ ReLU(a). The SNN prediction is the mean
    of the N per-timestep outputs. States reset between samples. N in
    {25, 50, 100, 200} to show convergence toward the ANN.
 4. Metrics: test R2 + spectral NRMSE for ANN vs SNN; drop quantified.

Also validates both conv trunks against Speck memory constraints with
sinabs.utils.validate_memory_mapping_speck.

Output: experiments/snn_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import (MODEL_BUILDERS, TRAIN_OUTLIER_ROW, eval_all, load_variant,
                    predict, save_json, train_model)

N_STEPS = (25, 50, 100, 200)


def reset_states(net):
    for m in net.modules():
        if hasattr(m, "reset_states"):
            m.reset_states()


def snn_predict(snn, sc, X, n_steps):
    """Constant-current rate encoding: present scaled frame for n_steps,
    average the n_steps outputs. One sample at a time, states reset."""
    preds = []
    Xs = sc.x(X)
    with torch.no_grad():
        for i in range(len(Xs)):
            reset_states(snn)
            x = torch.from_numpy(Xs[i]).unsqueeze(0).repeat(n_steps, 1, 1, 1).to(common.DEVICE)
            out = snn(x)                      # (n_steps, T_out)
            preds.append(out.mean(dim=0).cpu().numpy())
    return sc.y_inv(np.stack(preds))


def convert_and_eval(name, seed):
    from sinabs.from_torch import from_model

    build_fn, var = MODEL_BUILDERS[name]
    d = load_variant(var)
    keep = np.array([i for i in range(18) if i != TRAIN_OUTLIER_ROW])
    Xtr, ytr = d["X_train"][keep], d["y_train"][keep]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]
    fs = d["fs"]

    # For sinabs the network must see 4-D conv input; y128/full variants carry
    # a Reshape2D front — convert (B, C, T) to (B, C, H, W) before the net.
    model, sc, _, ep = train_model(build_fn, Xtr, ytr, Xva, yva, seed)
    ann_te = eval_all(yte, predict(model, sc, Xte), fs)

    # strip Reshape2D if present (sinabs wants pure conv stack); pre-reshape X
    layers = list(model.children())
    if layers and isinstance(layers[0], common.Reshape2D):
        h, w = layers[0].h, layers[0].w
        model = torch.nn.Sequential(*layers[1:])

        def pre(X):
            return X.reshape(len(X), X.shape[1], h, w)
    else:
        def pre(X):
            return X

    input_shape = pre(Xte).shape[1:]
    snn_net = from_model(model.cpu(), input_shape=tuple(input_shape),
                         batch_size=1, add_spiking_output=False)
    snn = snn_net.spiking_model.to(common.DEVICE)

    res = {"seed": seed, "epochs_run": ep, "ann_test": ann_te, "snn_test": {}}
    for n in N_STEPS:
        pte = snn_predict(snn, sc, pre(Xte), n)
        m = eval_all(yte, pte, fs)
        res["snn_test"][str(n)] = m
        print(f"[{name} SNN N={n:3d}] test R2={m['r2']:+.3f} "
              f"(ANN {ann_te['r2']:+.3f}) specNRMSE={m['spec_nrmse']:.3f} "
              f"(ANN {ann_te['spec_nrmse']:.3f})")
    best_n = max(res["snn_test"], key=lambda k: res["snn_test"][k]["r2"])
    res["drop_r2_at_bestN"] = ann_te["r2"] - res["snn_test"][best_n]["r2"]
    res["drop_spec_nrmse_at_bestN"] = res["snn_test"][best_n]["spec_nrmse"] - ann_te["spec_nrmse"]
    res["best_n_steps"] = int(best_n)
    return res


def validate_speck(configs):
    from sinabs.utils import validate_memory_mapping_speck
    out = {}
    for label, layers in configs.items():
        layer_results = []
        ok = True
        for (cin, cout, k, dim) in layers:
            try:
                msg = validate_memory_mapping_speck(
                    input_feature_size=cin, output_feature_size=cout,
                    kernel_size=(k, k), stride=(1, 1), padding=(1, 1),
                    input_dimension=dim, conv_2d=True)
                layer_results.append({"layer": f"conv{cin}->{cout}@{dim}", "pass": True,
                                      "msg": str(msg)})
            except Exception as e:
                ok = False
                layer_results.append({"layer": f"conv{cin}->{cout}@{dim}", "pass": False,
                                      "msg": str(e)})
        out[label] = {"all_pass": ok, "layers": layer_results}
        print(f"[speck-validate {label}] {'PASS' if ok else 'FAIL'}")
    return out


def main():
    try:
        import sinabs
        ver = sinabs.__version__
    except ImportError as e:
        save_json({"sinabs": "unavailable", "error": str(e)},
                  common.ROOT / "experiments" / "snn_results.json")
        print("sinabs unavailable; skipping. ", e)
        return

    fixed = json.load(open(common.ROOT / "experiments" / "fixed_split_results.json"))
    results = {"sinabs_version": ver, "models": {}}

    for name in ("speck3k5", "speck18k"):
        seed = fixed["models"][name]["best_seed"]
        print(f"\n=== {name} (best seed {seed}) ===")
        results["models"][name] = convert_and_eval(name, seed)

    results["speck_memory_validation"] = validate_speck({
        "speck3k5 (2ch 32x32)": [(2, 8, 3, (32, 32)), (8, 16, 3, (8, 8))],
        "speck18k (18ch 32x64)": [(18, 8, 3, (32, 64)), (8, 16, 3, (8, 16))],
    })

    save_json(results, common.ROOT / "experiments" / "snn_results.json")
    print("\nSaved experiments/snn_results.json")


if __name__ == "__main__":
    main()
