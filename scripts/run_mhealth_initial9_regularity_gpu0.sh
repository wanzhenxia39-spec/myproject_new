#!/usr/bin/env bash
set -euo pipefail
experiment_output=/data/wanzhen/experiments/mhealth_initial9_regularity_seed42
mkdir -p "$experiment_output/cache/tmp" "$experiment_output/cache/pycache" "$experiment_output/cache/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$experiment_output/cache/pycache"
export TMPDIR="$experiment_output/cache/tmp"
export MPLCONFIGDIR="$experiment_output/cache/matplotlib"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/myproject_new/scripts/run_mhealth_initial9_regularity.py "$@"
