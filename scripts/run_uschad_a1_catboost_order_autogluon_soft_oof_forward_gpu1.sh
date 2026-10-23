#!/usr/bin/env bash
set -euo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"

cd "$project_root"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

exec "$python_bin" \
  scripts/run_uschad_a1_autogluon_soft_oof_catboost_forward.py \
  train-test \
  --gpu-id 1 \
  --time-limit 300 \
  --presets medium_quality \
  --num-cpus 8 \
  --num-bag-folds 2 \
  --order-num-cpus 112 \
  --iterations 500 \
  --learning-rate 0.05 \
  --depth 6 \
  "$@"
