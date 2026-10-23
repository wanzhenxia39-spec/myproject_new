#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="${MHEALTH_ZSL_RUN_ROOT:-${project_root}/runtime/zshar_runs}"
time_limit="${MHEALTH_ZSL_TIME_LIMIT:-300}"
num_cpus="${MHEALTH_ZSL_NUM_CPUS:-8}"
poll_seconds="${MHEALTH_ZSL_GPU1_POLL_SECONDS:-60}"
lock_file="/tmp/wanzhen_myproject_new_gpu1_mhealth_zsl_autogluon_default.lock"
run_name_a1="mhealth_zsl_autogluon_default_ensemble_initial_a1_v1_5_independent_t${time_limit}_random20_eval30_seed42_gpu1_rerun"
run_name_a2="mhealth_zsl_autogluon_default_ensemble_a2_semantic_v3_independent_t${time_limit}_random20_eval30_seed42_gpu1_rerun"

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

exec 9>"${lock_file}"
echo "$(date '+%F %T') waiting for the GPU1 MHEALTH ZSL lock" >&2
flock 9
echo "$(date '+%F %T') acquired the GPU1 MHEALTH ZSL lock" >&2

gpu_uuid="$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
while true; do
  active="$({ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits || true; } \
    | awk -F',' -v uuid="${gpu_uuid}" '$1 == uuid {gsub(/ /, "", $2); print $2}' \
    | paste -sd, -)"
  if [[ -z "${active}" ]]; then
    break
  fi
  echo "$(date '+%F %T') waiting for physical GPU1; active compute PIDs=${active}" >&2
  sleep "${poll_seconds}"
done

a1_runner="scripts/run_mhealth_a1_autogluon_default_ensemble.py"
a2_runner="scripts/run_mhealth_a2_autogluon_default_ensemble.py"

"${python_bin}" -u "${a1_runner}" check \
  --gpu-id 1 --time-limit "${time_limit}" --num-cpus "${num_cpus}" \
  --run-root "${run_root}" --run-name "${run_name_a1}"

"${python_bin}" -u "${a2_runner}" check \
  --gpu-id 1 --time-limit "${time_limit}" --num-cpus "${num_cpus}" \
  --run-root "${run_root}" --run-name "${run_name_a2}"

"${python_bin}" -u "${a1_runner}" train-test \
  --gpu-id 1 --time-limit "${time_limit}" --num-cpus "${num_cpus}" \
  --run-root "${run_root}" --run-name "${run_name_a1}" --resume

exec "${python_bin}" -u "${a2_runner}" train-test \
  --gpu-id 1 --time-limit "${time_limit}" --num-cpus "${num_cpus}" \
  --run-root "${run_root}" --run-name "${run_name_a2}" --resume
