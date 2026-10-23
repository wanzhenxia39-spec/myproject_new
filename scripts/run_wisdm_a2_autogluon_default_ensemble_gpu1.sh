#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wanzhen/myproject_new
export PYTHONPATH=/home/wanzhen/myproject_new/src

exec /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_wisdm_a2_autogluon_default_ensemble.py \
  train-test \
  --gpu-id 1 \
  --time-limit 300 \
  --resume \
  "$@"
