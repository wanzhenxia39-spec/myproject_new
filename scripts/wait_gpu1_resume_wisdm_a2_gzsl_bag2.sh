#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs"
gpu_id=1
poll_seconds=60
lock_file="/tmp/wanzhen_wisdm_a2_gzsl_bag2_gpu1.lock"

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another WISDM GZSL GPU-1 waiter/run already holds ${lock_file}."
  exit 1
fi

gpu_has_compute_processes() {
  local gpu_uuid query
  if ! gpu_uuid="$(
    nvidia-smi --id="${gpu_id}" --query-gpu=uuid --format=csv,noheader 2>&1
  )"; then
    echo "$(date '+%F %T') Unable to query GPU ${gpu_id}: ${gpu_uuid}"
    return 0
  fi
  gpu_uuid="${gpu_uuid//[[:space:]]/}"
  if ! query="$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>&1
  )"; then
    echo "$(date '+%F %T') Unable to query GPU processes: ${query}"
    return 0
  fi
  grep -Fq "${gpu_uuid}," <<<"${query}"
}

echo "Waiting for physical GPU ${gpu_id} to have no compute processes..."
while gpu_has_compute_processes; do
  echo "$(date '+%F %T') GPU ${gpu_id} is busy; checking again in ${poll_seconds}s."
  sleep "${poll_seconds}"
done

echo "$(date '+%F %T') GPU ${gpu_id} is free; resuming WISDM GZSL from the existing run directory."
cd "${project_root}"
export PYTHONPATH="${project_root}/src"
exec "${python_bin}" -u \
  scripts/run_wisdm_a2_gzsl_autogluon_soft_oof_reuse_zsl_order.py \
  train-test \
  --gpu-id "${gpu_id}" \
  --num-cpus 8 \
  --time-limit 300 \
  --num-bag-folds 2 \
  --run-root "${run_root}" \
  --resume
