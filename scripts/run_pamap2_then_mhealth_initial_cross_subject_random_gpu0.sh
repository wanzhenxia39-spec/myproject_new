#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs"

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

exec "${python_bin}" -u \
  scripts/run_pamap2_mhealth_initial_cross_subject_autogluon_soft_oof.py \
  train-test \
  --dataset all \
  --attribute-order random \
  --random-order-seed 42 \
  --gpu-id 0 \
  --num-cpus 8 \
  --time-limit 300 \
  --num-bag-folds 2 \
  --run-root "${run_root}" \
  --resume
