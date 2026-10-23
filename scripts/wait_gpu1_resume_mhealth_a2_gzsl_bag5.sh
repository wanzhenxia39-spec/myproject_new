#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs"
gpu_id=1
poll_seconds=60
free_memory_threshold_mib=30000
lock_file="/tmp/wanzhen_mhealth_a2_gzsl_bag5_gpu1.lock"

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another MHEALTH GZSL GPU-1 waiter/run already holds ${lock_file}."
  exit 1
fi

gpu_free() {
  local used total query
  if ! query="$(
    nvidia-smi --id="${gpu_id}" \
      --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits 2>&1
  )"; then
    echo "$(date '+%F %T') Unable to query GPU ${gpu_id}: ${query}"
    return 1
  fi
  IFS=, read -r used total <<<"${query}"
  used="${used//[[:space:]]/}"
  total="${total//[[:space:]]/}"
  if [[ ! "${used}" =~ ^[0-9]+$ || ! "${total}" =~ ^[0-9]+$ ]]; then
    echo "$(date '+%F %T') Unexpected GPU status: ${query}"
    return 1
  fi
  (( total - used >= free_memory_threshold_mib ))
}

echo "Waiting for physical GPU ${gpu_id} to have at least ${free_memory_threshold_mib} MiB free..."
while ! gpu_free; do
  echo "$(date '+%F %T') GPU ${gpu_id} is busy; checking again in ${poll_seconds}s."
  sleep "${poll_seconds}"
done

echo "$(date '+%F %T') GPU ${gpu_id} is free; resuming MHEALTH GZSL."
cd "${project_root}"
export PYTHONPATH="${project_root}/src"
exec "${python_bin}" -u \
  scripts/run_mhealth_a2_gzsl_autogluon_soft_oof_reuse_order_bag5.py \
  train-test \
  --gpu-id "${gpu_id}" \
  --num-cpus 8 \
  --time-limit 300 \
  --run-root "${run_root}" \
  --resume
