# Non-Contact Vibration Signal Reconstruction from Event Camera Data with Lightweight CNNs for Condition Monitoring

Code and results for a paper under review at IFAC SAFEPROCESS 2027.

## Overview

An event camera watches a cantilever beam while a laser Doppler vibrometer (LDV) records its vibration velocity. We compress the raw megapixel event stream into a grid of per-region event rates and train a small 1D CNN (140K parameters) to reconstruct the LDV signal. On held-out trials the model reaches test R-squared 0.81 and preserves the spectral shape and damping behavior that matter for fault detection. Every learned result is reported against ridge regression on the same representation, under a leakage-audited protocol with cross-validation and seed-variance disclosure.

## Dataset

The experiments use the Beam subset of XJTU-DV (Li et al., 2026). The raw data is not redistributed here.

1. Download `XJTU-DV-Beam.zip` (1.66 GB) from Zenodo: https://zenodo.org/records/17447083 (DOI 10.5281/zenodo.17447083). Only the Beam archive is needed, not the full 21 GB record.
2. Extract it so the layout is:

```
data/XJTU-DV-Beam/
  DV/   Off_set1_trail1.dat ... (30 event recordings)
  LDV/  Off_set1_trail1.txt ... (30 vibrometer records)
```

Three derived feature files are included at the repository root so the training scripts run without the raw download: `processed_data.npz` (3x3 grid), `processed_data_grid5.npz` (5x5), and `processed_data_grid10.npz` (10x10, the headline representation). They are event-rate features computed from XJTU-DV-Beam and can be regenerated from the raw data (steps below).

## Setup

```
conda create -n beam python=3.12
conda activate beam
pip install -r requirements.txt
```

A CUDA-capable PyTorch build shortens training, but every run in the paper takes seconds to minutes per model and also works on CPU.

## Reproducing the results

Scripts live in `experiments/` and read/write relative to the repository root. The shared protocol (splits by trial, seeds 42 to 51, restart-on-collapse from validation only, warmup then cosine learning-rate schedule, early stopping on validation loss) is implemented in `experiments/common.py` and `experiments/run_seed_protocol.py`.

1. Data preparation (only needed if you want to rebuild the included npz files):
   - `python save_processed_data.py` builds the 3x3 representation (`processed_data.npz`).
   - `python experiments/run_grid_ablation.py` builds the 5x5 and 10x10 representations from raw data, verifies the pipeline against `processed_data.npz` bit-for-bit, and runs the grid-resolution ablation.
2. Headline results: `python experiments/run_headline_10x10.py` trains the 10x10 model on the headline split (test = trial 2) and the secondary split (test = trial 5), runs five-fold cross-validation by trial, and refits the ridge and mean-signal baselines on the same representation. Output: `experiments/headline_10x10_results.json`.
3. 3x3 reference numbers and the architecture comparison: `experiments/run_seed_protocol.py`, `experiments/run_arch_comparison.py`, `experiments/run_arch_small.py`.
4. Validity audits (scaler leakage, temporal alignment, split independence, trivial baselines): `audits/audit1_scaler_leakage.py` through `audit4_baselines.py`; findings summarized in `experiments/RESULTS.md`.
5. Comparison with the training-free REF-EvSet spectral baseline: clone https://github.com/haoqiu111/REFEvSet-V1 into `competitors/REFEvSet-V1`, build their beam level-1 cache per their `reproduce.sh` stage 1, run their `scripts/beam_calibration.py`, then `python experiments/run_refevset_comparison.py`. The paired comparison over all 30 recordings, using cross-validation fold predictions, is `python experiments/run_reviewer_fixes.py`. Their repository is not vendored here.
6. Figures: `python experiments/make_figures.py` regenerates the four paper figures from the saved results.

`experiments/RESULTS.md` collects the final numbers and points each one to the JSON file that carries the raw values.

## Results summary

Headline split: test = trial 2, validation = trial 4, training = trials 1, 3 and 5 minus one audit-excluded recording. Ten seed slots with restart-on-collapse; ridge is refit on the same input.

| Model (10x10 grid, 200 channels) | Test R-squared | Spectral NRMSE (0-40 Hz) | Envelope R-squared |
|---|---|---|---|
| 1D CNN (139.7K params) | 0.811 +/- 0.011 (95% CI [0.804, 0.817]) | 0.261 | 0.912 |
| Ridge on the same input | 0.667 | 0.300 | 0.780 |
| Per-timestep train mean | 0.060 | 0.711 | 0.306 |

Five-fold cross-validation by trial: 1D CNN 0.741 +/- 0.224 (95% CI [0.622, 0.844]) vs ridge 0.616; the CNN wins all five folds. Grid-resolution ablation (test R-squared): 1x1 collapses (0.06), 3x3 reaches 0.710, 5x5 reaches 0.737, 10x10 reaches 0.815. Against the training-free REF-EvSet baseline the model wins 27 of 30 recordings on time-domain R-squared (0.773 vs 0.644, sign test p < 0.0001) even though their reconstruction receives a per-recording amplitude and lag fit. Full tables, per-fold values and restart counts are in `experiments/RESULTS.md` and the result JSON files.

## Citation

The paper is under review at IFAC SAFEPROCESS 2027. A citation entry will be added here after the review. Until then please cite this repository.

## License

MIT, see `LICENSE`.

## Acknowledgements

The XJTU-DV dataset is by X. Li et al., "XJTU-DV: Open-source dynamic vision dataset for non-contact vibration measurement and fault diagnosis of mechanical systems", Chinese Journal of Mechanical Engineering 39:100163, 2026 (data: DOI 10.5281/zenodo.17447083). The REF-EvSet baseline used for comparison is by its respective authors (https://github.com/haoqiu111/REFEvSet-V1, MIT license).
