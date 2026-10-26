#!/usr/bin/env bash
set -euo pipefail
analysis_dataset="${1:-pamap2}"
if [ "$#" -gt 0 ]; then shift; fi
analysis_output="/home/wanzhen/attribute_prototype_analysis/new_results/initial9_zsl_loao_by_activity/$analysis_dataset"
mkdir -p "$analysis_output/cache/tmp" "$analysis_output/cache/pycache" "$analysis_output/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$analysis_output/cache/pycache"
export TMPDIR="$analysis_output/cache/tmp"
export MPLCONFIGDIR="$analysis_output/cache/matplotlib"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/attribute_prototype_analysis/analyze_initial9_loao_by_activity.py \
  --dataset "$analysis_dataset" --output "$analysis_output" "$@"
