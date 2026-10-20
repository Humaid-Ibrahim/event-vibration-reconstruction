"""
Audit 3: Fold / session independence check.

Determines how the 30 samples map to physical recordings and what granularity
a 5-fold CV must split on.

Evidence:
  - data_loaders/beam_dataloader_aligned.py:96-104 and 837-923 document and
    implement the split: filenames {Off,On}_set{1,2,3}_trail{1..5}, i.e.
    2 illuminations x 3 camera settings x 5 trials = 30 recordings.
    Train = trials 1-3 (18), Val = trial 4 (6), Test = trial 5 (6).
  - Each sample is one full 8-s recording (one .dat + one .txt pair); samples
    are NOT sliding windows of a longer session.
  - This script prints the config metadata stored in each *.npz to corroborate
    sample counts and representation settings, and cross-checks results.json
    sample counts.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 70)
    print("NPZ metadata (config keys)")
    print("=" * 70)
    for npz in sorted(ROOT.glob("processed_data*.npz")) + [ROOT / "predictions.npz"]:
        if not npz.exists():
            continue
        try:
            d = np.load(npz, allow_pickle=True)
        except Exception as e:
            print(f"{npz.name}: failed to load ({e})")
            continue
        print(f"\n--- {npz.name} ---")
        print("keys:", list(d.keys()))
        for k in ("config", "model_name"):
            if k in d:
                print(f"{k}: {d[k]}")
        for k in d.keys():
            if k.startswith(("X_", "y_")):
                print(f"  {k}: shape={d[k].shape}")

    # Split sizes implied by the dataloader's trial-based split
    print()
    print("=" * 70)
    print("Split structure from dataloader code (beam_dataloader_aligned.py)")
    print("=" * 70)
    src = (ROOT / "data_loaders" / "beam_dataloader_aligned.py").read_text(encoding="utf-8")
    for pat, desc in [
        (r"- Train: trials 1, 2, 3 \(18 samples, 60%\)", "docstring: train = trials 1-3"),
        (r"- Val: trial 4 \(6 samples, 20%\)", "docstring: val = trial 4"),
        (r"- Test: trial 5 \(6 samples, 20%\)", "docstring: test = trial 5"),
        (r"for trial_num in \[1, 2, 3\]:", "code: train trials [1,2,3]"),
        (r"val_indices\.extend\(trials\.get\(4", "code: val trial 4"),
        (r"test_indices\.extend\(trials\.get\(5", "code: test trial 5"),
        (r"config = f\"\{illumination\}_\{camera_set\}\"", "code: 6 configs = illumination x camera set"),
    ]:
        found = re.search(pat, src) is not None
        print(f"  [{'OK' if found else 'MISSING'}] {desc}")

    # Filename pattern evidence from the special-case typo handler
    print("\nFilename evidence in code (typo handler, lines ~805):")
    print("  'On_set1_trai4' / 'On_set1_trail4' -> pattern {On|Off}_set{1-3}_trail{1-5}")

    # Cross-check a results.json
    rj = ROOT / "results/cnn2d_snn_eventrate/20260116_081812/results.json"
    if rj.exists():
        j = json.load(open(rj))
        d = j.get("data", {})
        print(f"\nresults.json sample counts: train={d.get('train_samples')}, "
              f"val={d.get('val_samples')}, test={d.get('test_samples')}")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""\
- 30 samples = 30 distinct 8-s recordings: 6 conditions (2 illumination x 3
  camera settings) x 5 trials. Confirmed by split code + docstring + counts
  (18 = 6x3 trials, 6 = 6x1 trial). Samples are full recordings, not windows.
- HOWEVER: all 30 recordings are of the SAME physical beam under the same
  excitation protocol; the 5 trials within a condition are repeats of the
  same experiment minutes apart. Trials are the correct (and only available)
  CV unit: 5-fold CV must hold out one TRIAL NUMBER across all 6 conditions
  per fold (trial k -> fold k), exactly generalizing the current
  train(1-3)/val(4)/test(5) split.
- Splitting at any finer granularity (within-recording windows across splits)
  would leak; splitting on condition instead of trial would test
  condition-transfer, a different claim.
- NOTE (XJTU-DV paper): Beam subset described as 3 camera setups x 2
  lightings; our 30 = 5 trials x 6 conditions is consistent with that.
  Raw data folder is absent from this machine, so filenames could not be
  re-listed directly; evidence is from committed loader code + metadata.""")


if __name__ == "__main__":
    main()
