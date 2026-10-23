#!/usr/bin/env bash
set -euo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"

cd "$project_root"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

exec "$python_bin" \
  scripts/run_pamap2_a1_catboost_order_independent_backends.py \
  train-test \
  --gpu-id 1 \
  --backends catboost,logistic_regression \
  --num-cpus 8 \
  "$@"
