# Initial 9-attribute predictability on Seen IMU windows

Entry point: `evaluate_initial9_seen_predictability.py`. This analysis reuses the
existing **independent** AutoGluon ZSL models for PAMAP2, MHEALTH, USC-HAD, WISDM
and DSADS. Each dataset uses its 20 seed-42 class splits and the initial v1.5
9-dimensional activity attribute vectors. It does not use refined A2 vectors,
classifier chains, GZSL checkpoints, or unseen test predictions.

## Protocol

For each class split, reconstruct the project's original stratified 70/30 Seen
window split with `random_state=42`, retaining the original dataset row indices.
The outer 30% is the evaluation set used for these matrices.

The existing `AutoGluonBackend.fit` accepts `X_val` and `y_val` arguments but does
**not** pass them to `TabularPredictor.fit` as `tuning_data`. AutoGluon therefore
holds out an inner subset of the outer 70% for selecting models, ensemble
weights and any binary threshold calibration. The original outer validation
accuracy also supplies attribute weights for activity-prototype matching; this
analysis evaluates attribute predictions directly, before activity matching.

The evaluator checks every saved predictor's cached fitting/tuning row indices:

* Their union must equal the reconstructed outer training set.
* Neither may intersect the outer validation set.
* Their saved encoded labels must agree with the current v1.5 attribute targets.
* Replayed outer-validation Macro-F1 must match the saved validation report.

No predictor is refitted, recalibrated or selected on the new evaluation results.
Each nonconstant attribute uses its already saved best model and saved binary
decision threshold. Attributes constant in a Seen training split use the
original constant backend and are separately identified in the coverage report.

## Metrics

Each attribute uses the declared ordinal ontology `Low=0, Medium=1, High=2`.
Missing trained states receive probability zero; probability columns are aligned
by **label name** rather than by AutoGluon's internal numeric class order.

`EOD = mean_i sum_l q_i(l) * abs(l - target_i) / 2`.

The denominator remains 2 even when only one or two states occur in a split.
This is a probability-weighted expected distance, not the MAE of the hard
prediction or the error in the expected ordinal value.
Float32 ensemble probabilities can overshoot 1 by one ULP (about 1.19e-7).
Only out-of-range rows within a 1e-6 tolerance are clipped and renormalized;
materially invalid probabilities or non-normalized distributions are rejected.

The primary `macro_f1` averages F1 over the union of true and predicted states,
with `zero_division=0`, exactly as the project's existing validation reports do.
The supplementary `macro_f1_all_levels` always averages over all three declared
states, assigning zero to a completely absent state. State coverage is exported
so that absent states and constant attributes are not mistaken for evidence of
multi-state discrimination.

Each matrix entry is the **equal-weight mean of 20 split metrics**, not a pooled
score across repeated windows. Separate CSVs contain the sample standard
deviation (`ddof=1`) and split count. The class splits share windows; the standard
deviation describes split variation and is not an independent-sample error bar.
Training-majority Macro-F1 and training-prior EOD are exported as baselines.

## Outputs and limitations

All outputs, logs and caches are under:

`/data/wanzhen/experiments/attribute_predictability/initial9_zsl_seen_validation_seed42/`

Main results: `macro_f1_mean_9x5.csv` and `eod_mean_9x5.csv`, with attributes as
rows and datasets as columns. Heatmaps are available as PNG/PDF/SVG. An XLSX
workbook is also written when `openpyxl` is installed. Per-split predictions,
partitions, audits, metrics and input hashes support reproduction.

The attribute targets are the designed **activity-level semantic labels**
assigned to each activity's windows. These results measure predictability of
those labels from the existing IMU feature representation on held-out Seen
windows. They do not establish unseen-activity transfer, independent-subject
generalization, or direct physical ground truth for each individual window.
The existing window-level split can share subjects and overlapping recordings
between partitions. Keep this qualification when interpreting high scores.

## Run

```bash
OUT=/data/wanzhen/experiments/attribute_predictability/initial9_zsl_seen_validation_seed42
mkdir -p "$OUT/cache/tmp" "$OUT/cache/matplotlib"
CUDA_VISIBLE_DEVICES='' \
PYTHONPYCACHEPREFIX="$OUT/cache/pycache" \
MPLCONFIGDIR="$OUT/cache/matplotlib" TMPDIR="$OUT/cache/tmp" \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/myproject_new/scripts/evaluate_initial9_seen_predictability.py \
  --output "$OUT" --resume > "$OUT/evaluation.log" 2>&1
```

Inference runs on CPU and does not reserve either GPU. `--check` validates the
source run schemas and split coverage without inference. `--resume` reuses
completed, hash-matching split results. Use a separate output directory when
debugging a subset with `--datasets` or `--max-splits`.
