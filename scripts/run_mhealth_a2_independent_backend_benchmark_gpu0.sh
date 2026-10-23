#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

exec "${python_bin}" -u \
  scripts/run_mhealth_a2_independent_backend_benchmark.py \
  train-test \
  --gpu-id 0 \
  --backends catboost,xgboost,extra_trees,random_forest,lightgbm \
  --num-cpus 8 \
  --resume \
  "$@"
