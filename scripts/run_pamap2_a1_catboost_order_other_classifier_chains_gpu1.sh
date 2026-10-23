#!/usr/bin/env bash
set -euo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"

cd "$project_root"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

exec "$python_bin" \
  scripts/run_pamap2_a1_catboost_order_other_classifier_chains.py \
  train-test \
  --gpu-id 1 \
  --base-backends lightgbm,xgboost,linear_svm,extra_trees,random_forest,neuralnet_fastai,neuralnet_torch,ordinal_logistic,coral_mlp,mlp_independent \
  --num-cpus 8 \
  "$@"
