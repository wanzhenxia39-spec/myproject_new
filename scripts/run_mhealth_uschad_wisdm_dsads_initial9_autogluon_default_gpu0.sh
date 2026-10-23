#!/usr/bin/env bash
set -euo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="${ZSHAR_AUTOGLUON_PYTHON:-/home/wanzhen/miniconda3/envs/zara/bin/python}"
script="${project_root}/scripts/run_mhealth_uschad_wisdm_dsads_initial9_autogluon_default.py"
log_root="${ZSHAR_AUTOGLUON_LOG_ROOT:-/home/wanzhen/LLMbasedZeroshotHARAgent/zshar_logs/mhealth_uschad_wisdm_dsads_initial9_autogluon_default}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi
if [[ ! -f "${script}" ]]; then
  echo "Experiment script not found: ${script}" >&2
  exit 2
fi

mkdir -p "${log_root}"
log_file="${log_root}/run_$(date +%Y%m%d_%H%M%S).log"
queue_lock="${log_root}/gpu0_queue.lock"
python_path="${project_root}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  python_path="${python_path}:${PYTHONPATH}"
fi

# Capture the waiting phase and the later experiment phase in the same file.
exec > >(tee -a "${log_file}") 2>&1
echo "Log: ${log_file}"
echo "Sequence: MHEALTH -> USC-HAD -> WISDM -> DSADS; initial 9-D; physical GPU0"

# Prevent two copies of this four-dataset queue from waiting/running at once.
exec 9>"${queue_lock}"
if ! flock -n 9; then
  echo "Another four-dataset GPU0 queue already holds: ${queue_lock}" >&2
  exit 3
fi

wait_interval="${ZSHAR_GPU_WAIT_INTERVAL:-30}"
if ! [[ "${wait_interval}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ZSHAR_GPU_WAIT_INTERVAL must be a positive integer" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to wait for GPU0" >&2
  exit 2
fi

pamap_python_pattern="/home/wanzhen/myproject_new/scripts/[r]un_pamap2_initial9_autogluon_default_ensemble.py"
while pgrep -f "${pamap_python_pattern}" >/dev/null 2>&1; do
  echo "[$(date '+%F %T')] Waiting: PAMAP2 initial-9D experiment is still running on GPU0."
  sleep "${wait_interval}"
done

gpu0_apps() {
  nvidia-smi --id=0 \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null
}

while true; do
  current_apps="$(gpu0_apps)" || {
    echo "Failed to query GPU0 compute processes with nvidia-smi" >&2
    exit 2
  }
  if [[ -n "${current_apps}" ]]; then
    echo "[$(date '+%F %T')] Waiting: GPU0 still has compute processes:"
    echo "${current_apps}"
    sleep "${wait_interval}"
    continue
  fi

  # Require two empty observations to avoid starting during a short hand-off
  # between AutoGluon attribute fits.
  sleep 10
  stable_apps="$(gpu0_apps)" || {
    echo "Failed to recheck GPU0 compute processes with nvidia-smi" >&2
    exit 2
  }
  if [[ -z "${stable_apps}" ]]; then
    break
  fi
  echo "[$(date '+%F %T')] GPU0 became busy during the stability check:"
  echo "${stable_apps}"
  sleep "${wait_interval}"
done

echo "[$(date '+%F %T')] GPU0 is free; starting the four-dataset sequence."
PYTHONPATH="${python_path}" PYTHONUNBUFFERED=1 \
  "${python_bin}" -u "${script}" train-test \
  --gpu-id 0 \
  --time-limit 300 \
  --resume \
  "$@"
