# Results

Final numbers behind the paper. Each section names the JSON file that carries the raw values; the scripts that produced them are listed in the README.

Shared protocol for every learned number: splits by trial (30 recordings = 6 conditions x 5 trials), seeds 42 to 51 for the fixed splits and 42 to 44 for cross-validation and ablations, restart-on-collapse detected from validation only (validation R-squared below 0.3 restarts with a new seed, three attempts maximum), 5-epoch warmup to learning rate 5e-4 then cosine decay, early stopping with patience 15, model selection on validation only. One recording (On_set2_trail3) is excluded from training and test scoring: a pre-modeling audit flagged its event-LDV misalignment, realignment from raw data did not make it predictable at any offset, and its event stream is saturated at our grid resolution.

## Headline: 10x10 grid, test on a never-developed-against trial

Headline split: test = trial 2, validation = trial 4, training = trials 1, 3 and 5 minus the excluded recording. Trial 2 was never used in any development decision. The 1D CNN takes the 10x10 polarity-split event-rate grid (200 channels x 2048 bins) and has 139.7K parameters. Ridge regression is refit on the same flattened input. Raw values: `headline_10x10_results.json`.

| Model | Test R2 | Spectral NRMSE (0-40 Hz) | Decay error (1/s) | Envelope R2 |
|---|---|---|---|---|
| 1D CNN | 0.811 +/- 0.011 (95% CI [0.804, 0.817]) | 0.261 | 0.215 | 0.912 |
| Ridge, same input | 0.667 | 0.300 | 0.120 | 0.780 |
| Per-timestep train mean | 0.060 | 0.711 | 0.120 | 0.306 |

No seed slot needed a restart. The predicted spectra carry no second-harmonic artifact: the 2f0/f0 energy ratio is 0.078 against 0.068 in the ground truth. On the secondary split (test = trial 5, training = trials 1 to 3 minus the excluded recording) the CNN reaches 0.723 +/- 0.027 against ridge at 0.660; the smaller margin on that split is in the JSON and the paper discusses it.

## Five-fold cross-validation by trial

Each fold holds out one trial as test and the next as validation, so every fold contains all six conditions in every role. Fold 3 is scored with the excluded recording removed, for the CNN and ridge alike. Raw values: `headline_10x10_results.json` (key `cv`).

Pooled over folds and seed slots the CNN reaches 0.741 +/- 0.224 (95% CI [0.622, 0.844]) against ridge at 0.616, and wins all five folds. Per-fold means: 0.31, 0.78, 0.91, 0.88, 0.84 for the CNN; 0.17, 0.64, 0.87, 0.67, 0.73 for ridge. Two of fifteen slots restarted; none remained collapsed. Trial 1 is the hard fold for both methods, so trial-to-trial heterogeneity rather than seed noise dominates the pooled variance.

## Grid-resolution and polarity ablation

Same headline split, three seed slots per setting, the same CNN with only the first convolution adapting to the channel count. Raw values: `grid_ablation_results.json`.

| Input | Channels | Test R2 |
|---|---|---|
| 1x1 (global rate) | 2 | 0.06 (collapses to the mean predictor in all nine attempts) |
| 3x3, polarity summed | 9 | 0.627 +/- 0.026 |
| 3x3, polarity split | 18 | 0.710 +/- 0.010 |
| 5x5, polarity split | 50 | 0.737 +/- 0.021 |
| 10x10, polarity split | 200 | 0.815 +/- 0.007 |

Spatial resolution improves every metric monotonically through 10x10, and the polarity split is worth about 0.08 R2 at 3x3. The parameter growth (34.9K to 139.7K) sits entirely in the first convolution, so the gain is a representation effect rather than a capacity effect.

## Model selection at the 3x3 representation

Six architectures under the identical protocol on the headline split, each family minimally tuned (at most two dropout or learning-rate variants, chosen on validation with three slots). Raw values: `arch_comparison_results.json`, `arch_small_results.json`, `seed_protocol_results.json`.

| Model | Params | Test R2 |
|---|---|---|
| 1D CNN (stride-1 conv stack) | 34.9K | 0.704 +/- 0.015 |
| Wider 1D CNN | 39K / 261K | 0.701 +/- 0.019 / 0.694 +/- 0.038 |
| 1D conv + biLSTM | 40K / 214K | 0.653 +/- 0.011 / 0.684 +/- 0.042 |
| 1D transformer encoder | 58K / 191K | 0.554 +/- 0.025 / 0.574 +/- 0.025 |
| Dilated TCN | 49K / 185K | 0.241 +/- 0.172 / 0.240 +/- 0.327 |
| 2D CNN, global bottleneck head | 3.13M | failed to train (every attempt collapsed to the mean predictor) |

Within every family the small-to-large gap is smaller than the gaps between families. The inductive bias (stride-1, translation-equivariant convolution with a per-timestep output) sets performance; parameter count does not.

## Comparison with REF-EvSet's training-free baseline

We ran the released REF-EvSet beam pipeline unmodified on our recordings and scored both methods with the same metric code on the six trial-5 recordings, on a common onset-relative 6 s frame. Their reconstruction has arbitrary units and no fixed alignment, so it receives a per-recording oracle fit (least-squares amplitude and a lag search against the LDV target); our model predicts calibrated velocity with no per-recording fitting. Raw values: `refevset_comparison.json` and `headline_10x10_results.json` (key `refevset_frame_ours10x10`).

Despite the asymmetry in their favor, our 10x10 model wins five of six recordings on time-domain R2 (median 0.86 against about 0.64) and has lower spectral NRMSE (0.347 against 0.497). The sixth recording scores below zero in this frame for our model (its full-window score is normal); the per-recording table is in the JSON. With six recordings a sign test cannot reach significance (p = 0.22 at 5 of 6). Neither method shows the 2x-f0 doubling artifact on any of the 30 recordings: both use signed event rates.

## Validity audits

Before any training we audited the pipeline (`audits/`, raw values in `scaler_audit.json`, `alignment_results.json`, `baseline_metrics.json`). The input and output scalers are fit on training data only, verified from code and from the fitted scaler statistics. Event-LDV alignment is consistent across 29 of 30 recordings (median lag 12 ms); the one outlier is the excluded recording described above. The 30 samples are 30 distinct physical recordings, so splitting by trial number is the correct cross-validation unit. Trivial baselines calibrated the task: the per-timestep training-mean waveform alone reaches validation R2 near 0.2 because repeated trials are phase-locked, which is why every learned result here is gated on beating ridge regression rather than the mean predictor.
