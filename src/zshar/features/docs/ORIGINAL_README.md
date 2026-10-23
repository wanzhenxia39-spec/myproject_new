# ZSHAR unified pipeline

`/home/wanzhen/myproject` is now the source-of-truth for the zero-shot HAR code. Dataset-specific Python pipelines have been replaced by one package and one command. `/home/wanzhen/LLMbasedZeroshotHARAgent` is the default runtime root and contains experiment logs, summaries and saved models.

## Layout

```text
myproject/
├── configs/
│   ├── datasets/       # one profile per dataset; paths, labels and defaults only
│   ├── attributes/     # class-level semantic vectors
│   └── metadata/       # sensor metadata used by the LLM planner
├── split/              # ZSL/GZSL/sample and cross-subject manifests
├── src/zshar/
│   ├── datasets/       # profile registry
│   ├── data.py         # shared npy loader
│   ├── splits.py       # activity/subject splits and evaluation sampling
│   ├── semantics/      # GAP, planner, vector completion and prototypes
│   ├── models/         # all 14 shared attribute-classifier backends
│   ├── evaluation/     # scoring, ZSL, GZSL and calibration
│   ├── rerank/         # no-op interface and shared LLM top-k helper
│   └── pipeline/       # training and experiment orchestration
├── tests/
└── legacy/             # preserved historical implementation
```

## Install and inspect

```bash
cd /home/wanzhen/myproject
/home/wanzhen/miniconda3/envs/zara/bin/python -m pip install -e .
zshar datasets
zshar backends
```

Without installing, replace `zshar` with `PYTHONPATH=src python -m zshar.cli`.

Supported profiles are `pamap2`, `mhealth`, `dsads`, `uschad`, and `wisdm`. Supported shared backends are AutoGluon, LightGBM, XGBoost, CatBoost, logistic regression, linear SVM, ExtraTrees, random forest, fastai/PyTorch neural nets, ordinal logistic, CORAL MLP, independent MLP and multitask MLP.

## Run experiments

One command changes datasets through `--dataset`:

```bash
zshar run --dataset pamap2 --scenario zsl --backend extra_trees
zshar run --dataset mhealth --scenario zsl --backend extra_trees
zshar run --dataset dsads --scenario gzsl --backend catboost --gamma 0.1
zshar run --dataset pamap2 --scenario cross_subject --backend extra_trees
```

`cross_subject` loads the dataset's index-explicit
`split/subject_80_20/<dataset>_subject80_20_balanced_seed<seed>.json`. Attribute
models see only train-subject windows; activity-prototype evaluation sees only
test-subject windows. `--scenario subject` is an alias.

For leakage-free automatic GZSL calibration, use outer-seen-only pseudo-unseen
class folds:

```bash
zshar run \
  --dataset pamap2 \
  --scenario gzsl \
  --backend extra_trees \
  --prototype-metric ordinal_probability \
  --attribute-weighting validation_accuracy \
  --calibration-mode tpe \
  --calibration-class-folds 3 \
  --calibration-tpe-trials 50
```

Each calibration fold withholds complete outer-seen activities as
pseudo-unseen, holds out separate pseudo-seen windows, trains a temporary
attribute model only on the remaining pseudo-seen rows, and selects temperature
by mean validation AUSUC. Gamma is selected from the aggregated fold curve with
seen-retention and one-standard-error safeguards. The final seen/unseen test
rows are never used for parameter selection. Use `--calibration-mode grid` when
Optuna is unavailable, or `fixed` with explicit `--temperature/--gamma`.

The existing LETS-GZSL-style path remains the default and applies
temperature-softmax before subtracting gamma from seen-class scores. To test
calibrated stacking alone on the original compatibility-score scale, add:

```bash
--gzsl-bias-correction seen_penalty_only
```

In this mode no softmax or temperature search is performed, and temperature is
fixed to `1.0` as an explicit sentinel in saved results. Gamma must be tuned
again because it is now measured in raw-score units; do not reuse gamma from a
temperature-softmax run. `scripts/run_pamap2_gzsl_seen_penalty_only_pilot.sh`
runs 14 attribute-classifier backends sequentially over the 20 fixed PAMAP2
GZSL splits (all registered backends except `autogluon_multilabel`), using the
`gzsl_20` seen20/unseen20 manifests and leakage-free pseudo-unseen calibration.
Temperature scaling is disabled by
default and can be enabled with `PAMAP2_GZSL_USE_TEMPERATURE_SCALING=1`.

The previous PAMAP2 AutoGluon bag-5, one-stack-level, 3600-second setup is:

```bash
CUDA_VISIBLE_DEVICES=0 zshar run \
  --dataset pamap2 \
  --scenario zsl \
  --backend autogluon \
  --time-limit 3600 \
  --num-bag-folds 5 \
  --num-stack-levels 1 \
  --num-gpus 1 \
  --prototype-metric ordinal_probability \
  --attribute-weighting validation_accuracy \
  --run-name pamap2_zsl_autogluon_bag5_stack1_time3600_seed42
```

`--time-limit` follows the historical behavior and is applied to each non-constant attribute classifier.

### Correlated AutoGluon multilabel backend

Use `autogluon_multilabel` to keep one AutoGluon predictor per attribute while
conditioning each predictor on the earlier predicted attributes:

```bash
CUDA_VISIBLE_DEVICES=0 zshar run \
  --dataset pamap2 \
  --scenario zsl \
  --backend autogluon_multilabel \
  --time-limit 600 \
  --prototype-metric ordinal_probability \
  --attribute-weighting validation_accuracy \
  --run-name pamap2_zsl_autogluon_multilabel_seed42
```

`consider_labels_correlation` defaults to true. The ten base attributes use a
fixed high-to-low predictability order derived from existing unseen-split
results; both the order and correlation switch can be overridden through
`--backend-config`. See
[`docs/AUTOGLUON_MULTILABEL_EXPERIMENT.md`](docs/AUTOGLUON_MULTILABEL_EXPERIMENT.md)
for the order, implementation semantics, controls, and matched pilot design.

### Soft AutoGluon bagging-OOF chain

Use `autogluon_soft_oof` to pass cumulative class probabilities and normalized
predictive entropy to later attribute predictors. Training uses AutoGluon's
native bagging OOF probabilities; inference uses `predict_proba`. Hard
attribute predictions are never chained, and `--num-bag-folds` must be at
least 2. This is a separate backend, so the independent `autogluon` and the
hard-chain `autogluon_multilabel` behavior remain unchanged.

The PAMAP2 strict-ZSL A2 scripts are:

```bash
./scripts/train_pamap2_zsl_autogluon_soft_oof.sh
./scripts/test_pamap2_zsl_autogluon_soft_oof.sh
```

See
[`docs/AUTOGLUON_SOFT_OOF_EXPERIMENT.md`](docs/AUTOGLUON_SOFT_OOF_EXPERIMENT.md)
for the exact feature construction, uncertainty definition, serialization,
split-specific A2 handling, and runtime controls.

### Soft CatBoost OOF chain

Use `catboost_soft_oof` to train a separate CatBoost classifier for each
attribute while passing stratified out-of-fold class probabilities and
normalized entropy to later classifiers. The retained full-data models produce
the corresponding probability features during validation and inference. The
existing hard-label `catboost_classifier_chain` backend is unchanged.

The matched PAMAP2 strict-ZSL A2 launcher is:

```bash
./scripts/run_pamap2_catboost_soft_oof.sh
```

To restrict AutoGluon to the five requested level-1 models—CatBoost,
LightGBM, ExtraTrees, ZSHAR's custom independent MLP adapter, and logistic
regression—use:

```bash
CUDA_VISIBLE_DEVICES=0 zshar run \
  --dataset pamap2 \
  --scenario zsl \
  --backend autogluon \
  --autogluon-model-suite zshar_five \
  --time-limit 600 \
  --num-bag-folds 0 \
  --num-stack-levels 0 \
  --num-gpus 1 \
  --prototype-metric ordinal_probability \
  --attribute-weighting validation_accuracy \
  --run-name pamap2_zsl_autogluon_zshar_five_ordinal_probability_seed42
```

The five entries are the level-1 base-model types. Unless
`--no-weighted-ensemble` is specified, AutoGluon additionally fits its normal
weighted ensemble over their validation predictions. Per-model overrides are
accepted through `--backend-config`, for example
`'{"model_suite_config":{"mlp_independent":{"epochs":100}}}'`.

Outputs default to:

```text
/home/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs/<dataset>/<run-name>/
```

Each split contains `models/`, `metrics.json`, `details.csv` and `split.json`; the run root contains an incrementally updated `summary.csv`, exact config and dataset-profile snapshots. Use `--dry-run` to validate manifests and paths without training. In GZSL, sampled seen evaluation windows are explicitly removed from the classifier training frame.

Useful switches:

- `--split-indices 1,13,42` selects manifest indices.
- `--split-manifest` and `--sample-manifest` override the dataset defaults.
- `--subject-manifest` overrides the cross-subject 80/20 manifest.
- `--reuse-models` loads a completed split's model-manager manifest.
- `--backend-config '{"n_estimators":200}'` passes backend-specific settings.
- `--gamma-grid=-0.5,-0.25,0,0.25,0.5` reports a GZSL seen/unseen curve and AUSUC.

## GAP and semantics

GAP is dataset-independent and operates on a split plus semantic prototypes:

```bash
zshar gap \
  --dataset mhealth \
  --top-k 2 \
  --distance-metric mad \
  --output /home/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs/mhealth_gap.csv
```

The output contains `D_kNN`, `U_NN`, `U_Conf`, `JointGap_DkNN_UConf` and easy/medium/hard tertiles. LLM semantic planning and completion remain separate, reusable modules:

```bash
python -m zshar.semantics.planner --help
python -m zshar.semantics.completion --help
```

Live LLM calls read `LLM_API_KEY` or `GEMINI_API_KEY`; no credential is stored in source.

## Compatibility boundary

The unified runner covers classifier-to-prototype ZSL, strict index-explicit
cross-subject evaluation, fixed or outer-seen-only grid/TPE GZSL calibration,
gamma curves/AUSUC, and every existing classifier backend. Historical
feature-evidence and LLM reranking helpers are preserved under `zshar.rerank`
and `legacy/`; the old highly specialized shell scripts are archived for result
reproduction rather than treated as the maintained API.
