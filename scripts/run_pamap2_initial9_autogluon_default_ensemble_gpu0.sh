#!/usr/bin/env bash
set -euo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="${ZSHAR_AUTOGLUON_PYTHON:-/home/wanzhen/miniconda3/envs/zara/bin/python}"
script="${project_root}/scripts/run_pamap2_initial9_autogluon_default_ensemble.py"
log_root="${ZSHAR_AUTOGLUON_LOG_ROOT:-/home/wanzhen/LLMbasedZeroshotHARAgent/zshar_logs/pamap2_initial9_autogluon_default_ensemble}"

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
python_path="${project_root}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  python_path="${python_path}:${PYTHONPATH}"
fi

echo "Log: ${log_file}"
echo "PAMAP2 initial 9-D; AutoGluon default ensemble; physical GPU0"
PYTHONPATH="${python_path}" PYTHONUNBUFFERED=1 \
  "${python_bin}" -u "${script}" train-test \
  --gpu-id 0 \
  --time-limit 300 \
  --resume \
  "$@" 2>&1 | tee -a "${log_file}"
