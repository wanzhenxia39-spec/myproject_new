#!/usr/bin/env bash
set -euo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"

gpu_id=0
forwarded_args=()
while (($#)); do
  case "$1" in
    --gpu-id)
      if [[ $# -lt 2 ]]; then
        echo "--gpu-id requires a nonnegative integer" >&2
        exit 2
      fi
      gpu_id="$2"
      shift 2
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "Invalid --gpu-id: $gpu_id" >&2
  exit 2
fi

cd "$project_root"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpu_id"

exec "$python_bin" \
  scripts/run_wisdm_a1_autogluon_soft_oof_catboost_forward.py \
  train-test \
  --gpu-id "$gpu_id" \
  --time-limit 300 \
  --presets medium_quality \
  --num-cpus 8 \
  --num-bag-folds 2 \
  --order-num-cpus 112 \
  --iterations 500 \
  --learning-rate 0.05 \
  --depth 6 \
  "${forwarded_args[@]}"
