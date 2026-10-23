#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"

cd "${project_root}"
export PYTHONPATH="${project_root}/src"
export PYTHONUNBUFFERED=1

exec "${python_bin}" -u \
  scripts/run_mhealth_a2_autogluon_soft_oof_catboost_forward.py \
  train-test \
  --gpu-id 0 \
  --time-limit 300 \
  --presets medium_quality \
  --num-cpus 8 \
  --num-bag-folds 5 \
  --reuse-orders-only \
  --resume \
  "$@"
