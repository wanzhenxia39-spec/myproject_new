#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wanzhen/myproject_new
export PYTHONPATH=/home/wanzhen/myproject_new/src

exec /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_pamap2_a2_independent_backend_benchmark.py \
  train-test \
  --gpu-id 1 \
  --num-cpus 8 \
  --resume \
  "$@"
