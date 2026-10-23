#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs"
gpu_id=0
poll_seconds=60
required_idle_checks=2
lock_file="/tmp/wanzhen_pamap2_a2_gzsl_autogluon_default_gpu0.lock"

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another PAMAP2 GZSL AutoGluon-default GPU-0 waiter/run holds ${lock_file}."
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
idle_checks=0
while (( idle_checks < required_idle_checks )); do
  if gpu_has_compute_processes; then
    idle_checks=0
    echo "$(date '+%F %T') GPU ${gpu_id} is busy; checking again in ${poll_seconds}s."
  else
    ((idle_checks += 1))
    echo "$(date '+%F %T') GPU ${gpu_id} idle confirmation ${idle_checks}/${required_idle_checks}."
  fi
  if (( idle_checks < required_idle_checks )); then
    sleep "${poll_seconds}"
  fi
done

echo "$(date '+%F %T') GPU ${gpu_id} is free; starting PAMAP2 GZSL AutoGluon default ensemble."
cd "${project_root}"
export PYTHONPATH="${project_root}/src"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
exec "${python_bin}" -u \
  scripts/run_pamap2_a2_gzsl_autogluon_default_ensemble.py \
  train-test \
  --gpu-id 0 \
  --num-cpus 8 \
  --time-limit 300 \
  --run-root "${run_root}" \
  --resume
