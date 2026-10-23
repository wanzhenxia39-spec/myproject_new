#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
gpu_id="${GPU_ID:-1}"

cd "${project_root}"
export PYTHONPATH="${project_root}/src"

exec "${python_bin}" -u \
  scripts/run_mhealth_a2_gzsl_autogluon_soft_oof_catboost_forward.py \
  train-test \
  --gpu-id "${gpu_id}" \
  --time-limit 300 \
  --num-cpus 8 \
  --resume \
  "$@"
