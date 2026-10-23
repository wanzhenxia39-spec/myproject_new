#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="${MHEALTH_LEGACY_ZSL_RUN_ROOT:-${project_root}/runtime/zshar_runs}"
time_limit="${MHEALTH_LEGACY_ZSL_TIME_LIMIT:-300}"
num_cpus="${MHEALTH_LEGACY_ZSL_NUM_CPUS:-8}"
poll_seconds="${MHEALTH_LEGACY_ZSL_GPU1_POLL_SECONDS:-60}"
# Share the lock with the new-attribute MHEALTH queue so both jobs cannot race
# for physical GPU1 when they are launched independently.
lock_file="/tmp/wanzhen_myproject_new_gpu1_mhealth_zsl_autogluon_default.lock"
runner="scripts/run_mhealth_legacy_a1_a2_autogluon_default_ensemble.py"

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

common_args=(
  --gpu-id 1
  --time-limit "${time_limit}"
  --num-cpus "${num_cpus}"
  --run-root "${run_root}"
)

"${python_bin}" -u "${runner}" check --stage a1 "${common_args[@]}"
"${python_bin}" -u "${runner}" check --stage a2 "${common_args[@]}"

"${python_bin}" -u "${runner}" train-test --stage a1 \
  "${common_args[@]}" --resume

exec "${python_bin}" -u "${runner}" train-test --stage a2 \
  "${common_args[@]}" --resume
