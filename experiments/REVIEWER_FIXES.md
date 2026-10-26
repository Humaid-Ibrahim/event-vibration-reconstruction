# Reviewer fixes: n=30 comparison and decay diagnosis

Raw numbers: `reviewer_fixes_results.json`. Predictions: `cv_predictions_10x10.npz`.

## Task 1: REF-EvSet comparison at n=30

Each recording is scored with a model prediction from the cross-validation
fold in which it was a test item (10x10 representation, unified protocol,
best-validation run per fold). REF-EvSet rows are their training-free
reconstruction with per-recording lag and scale fitted against the LDV
reference, as before. Identical 6 s onset frame at 256 Hz.

| | n | ours R2 | REF R2 | R2 wins | sign p | Wilcoxon p | NRMSE wins | sign p |
|---|---|---|---|---|---|---|---|---|
| All recordings | 30 | 0.773 | 0.644 | 27/30 | <0.0001 | 0.0006 | 28/30 | <0.0001 |
| Excluding On_set2_trail3 | 29 | 0.800 | 0.642 | 27/29 | <0.0001 | <0.0001 | 28/29 | <0.0001 |

The previous n=6 comparison (p=0.22) is superseded: at n=30 the advantage is
statistically significant on both time-domain R2 and spectral NRMSE.

## Task 2: decay-rate gap diagnosis (main split, test = trial 2)

Signed decay-slope error (prediction minus truth; negative = over-damped):

| | mean | std |
|---|---|---|
| model | -0.231 | 0.102 |
| ridge | -0.084 | 0.061 |

The model is systematically over-damped, not merely noisy (bias exceeds
spread). Ridge shares the direction with smaller magnitude.

Smoothing ablation (model prediction):

| variant | decay err | time R2 | spec NRMSE | env R2 |
|---|---|---|---|---|
| raw | 0.231 | 0.824 | 0.240 | 0.922 |
| moving avg 3 | 0.231 | 0.827 | 0.237 | 0.921 |
| moving avg 5 | 0.229 | 0.830 | 0.235 | 0.918 |
| Savitzky-Golay w11 p2 (envelope) | 0.205 | 0.824 | 0.240 | 0.922 |
| Savitzky-Golay w11 p3 (envelope) | 0.205 | 0.824 | 0.240 | 0.922 |

Verdict: no smoothing variant closes the gap. The error is a systematic
over-damping bias, so the honest fix is disclosure, not post-processing.
Probable cause: MSE training under-weights the low-amplitude ringdown tail,
so predictions decay slightly faster than the truth. This goes in the
discussion as a stated limitation with the bias direction and magnitude.

Note: the task-1 CV predictions give slightly different per-recording values
than the earlier fold aggregates because predictions come from the best
validation run per fold rather than averaging three runs.
