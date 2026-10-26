# Initial 9-D ZSL leave-one-attribute-out

Run `evaluate_initial9_zsl_loao.py` to reuse the five datasets' existing seed-42,
initial-v1.5, independent AutoGluon classifiers. The existing 20 class splits,
fixed Unseen test window IDs, candidate order and original Seen-validation
accuracy weights remain fixed. No classifier is retrained, selected or
calibrated. No test labels enter matching weights or probability predictions.

For each split the script obtains each attribute's test probabilities once,
then computes the original full-9D prediction and nine leave-one-coordinate-out
predictions. Both the probability coordinate and the candidate prototype
coordinate are omitted from that variant's matching sum. Its denominator is
the sum of the **remaining** original weights.

With Low/Medium/High = 0/0.5/1:

`D_j(x,c) = sum_l p_j(l|x) * abs(level_l - prototype[c,j])`

`D_without_k(x,c) = sum_{j!=k} w_j D_j(x,c) / sum_{j!=k} w_j`

The score is `1 - D`, exactly as in the project evaluator. All probability
normalization uses the saved predictor's native class-column order to preserve
the original floating point arithmetic. The original candidate order and
`np.argsort(-scores)` tie handling are also preserved. Vectorized scores are
cross-checked against `rank_activity_prototypes` for all ten variants.

The full-9D predictions must match the historical `details.csv` on **every test
window**, and Top-1 and activity Macro-F1 must agree with the historical metrics.
The code stops on a mismatch rather than silently substituting a new baseline.

## Metrics and reporting

For each metric separately:

`delta(split,k) = metric(full9, split) - metric(without_k, split)`

Each final matrix cell averages those 20 **paired** differences. Positive means
the removal hurts; negative means removal improves the frozen system. The
tables/plots report **percentage points**, not relative percentages. The
absolute metric tables report percentages.

Macro-F1 here concerns final **activity classification**, not the preceding
attribute-prediction quality analysis. Every fixed Unseen test class is in the
truth set, so all candidate classes enter Macro-F1. All test windows are scored.
These balanced test manifests also make Top-1 equal to mean per-activity recall.

The two five-dataset horizontal bar charts show the 9 attributes in their actual
ontology order with a positive/negative zero axis. Error bars are sample standard
deviations of the 20 paired differences (`ddof=1`), not confidence intervals.
The shared class splits reuse windows; do not treat them as independent trials.
There are also individual two-panel figures for each dataset. PNG/PDF/SVG are
written for export.

An attribute with zero/negative delta is retained in the report. This is a
conditional marginal-contribution analysis of the frozen matching system,
not a guarantee that every attribute is necessary. Do not use these test deltas
to select a reduced set and then report that same test as an independent final
test of the selected set.

## Outputs

All test outputs, cached probabilities, logs and temporary files are on the data
disk under:

`/data/wanzhen/experiments/attribute_ablation/initial9_zsl_loao_seed42/`

* `initial9_zsl_loao_results.xlsx`: both metrics' full/ablated results,
  paired delta means/standard deviations, and all split-level metrics.
* `top1_delta_mean_pp_9x5.csv`, `macro_f1_delta_mean_pp_9x5.csv`: main 9x5 matrices.
* Corresponding `*_std_pp_9x5.csv`: paired-difference standard deviations.
* `top1_full_and_ablated_percent.csv`, `macro_f1_full_and_ablated_percent.csv`:
  full 9D plus nine removal variants as 10x5 tables.
* `ablation_summary.csv`, `per_split_metrics.csv`, `REPORT_zh.md`.
* `loao_top1_impact_5datasets.*`, `loao_macro_f1_impact_5datasets.*` and
  `<dataset>/loao_impact.*`: horizontal bar charts.
* Each `<dataset>/split_<index>/` contains `test_probabilities.npz`, a human-readable
  `test_probabilities.csv.gz`, ten variants' `activity_predictions.csv.gz`,
  `matching.json` with original and renormalized weights and candidate prototypes,
  a probability input/hash manifest, and `metrics.json`.

The probability cache permits further matching-only analyses without loading
AutoGluon or retraining. Its source signature and file hash are checked on reuse.

## Run

```bash
OUT=/data/wanzhen/experiments/attribute_ablation/initial9_zsl_loao_seed42
mkdir -p "$OUT/cache/tmp" "$OUT/cache/matplotlib"
CUDA_VISIBLE_DEVICES='' \
PYTHONPYCACHEPREFIX="$OUT/cache/pycache" \
MPLCONFIGDIR="$OUT/cache/matplotlib" TMPDIR="$OUT/cache/tmp" \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/myproject_new/scripts/evaluate_initial9_zsl_loao.py \
  --output "$OUT" --resume > "$OUT/evaluation.log" 2>&1
```

Inference runs on CPU. `--check` validates all source models, vectors and split
manifests without inference. `--resume` reuses completed/hash-matching splits.
`--summarize-only` regenerates tables and figures from the completed metric files.
XLSX export uses the existing local `openpyxl` dependency from the preceding
predictability analysis, when available.
