#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wanzhen/myproject_new
PYTHON=/home/wanzhen/miniconda3/envs/zara/bin/python
RUNNER="$PROJECT/scripts/run_pamap2_a1_gzsl_autogluon_default_ensemble_level_ablation_gpu0.py"

cd "$PROJECT"
export PYTHONPATH="$PROJECT/src"
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%F %T')] Preflight: PAMAP2 2-level GZSL"
"$PYTHON" -u "$RUNNER" check --level-count 2 --gpu-id 0 --time-limit 300 --num-cpus 8

echo "[$(date '+%F %T')] Training/evaluation: PAMAP2 2-level GZSL"
"$PYTHON" -u "$RUNNER" train-test --level-count 2 --gpu-id 0 --time-limit 300 --num-cpus 8 --resume

echo "[$(date '+%F %T')] Preflight: PAMAP2 5-level GZSL"
"$PYTHON" -u "$RUNNER" check --level-count 5 --gpu-id 0 --time-limit 300 --num-cpus 8

echo "[$(date '+%F %T')] Training/evaluation: PAMAP2 5-level GZSL"
"$PYTHON" -u "$RUNNER" train-test --level-count 5 --gpu-id 0 --time-limit 300 --num-cpus 8 --resume

echo "[$(date '+%F %T')] Completed both PAMAP2 GZSL level-ablation runs"
