#!/usr/bin/env bash
set -euo pipefail
experiment_output=/data/wanzhen/experiments/attribute_ablation/refined_v4_zsl_loao_seed42
mkdir -p "$experiment_output/cache/pycache" "$experiment_output/cache/matplotlib" "$experiment_output/cache/tmp"
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPYCACHEPREFIX="$experiment_output/cache/pycache"
export MPLCONFIGDIR="$experiment_output/cache/matplotlib"
export TMPDIR="$experiment_output/cache/tmp"
exec /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/attribute_prototype_analysis/reinfer_refined_v4_zsl_loao.py \
  --output "$experiment_output" --workers 5 --threads 2 --resume "$@"
