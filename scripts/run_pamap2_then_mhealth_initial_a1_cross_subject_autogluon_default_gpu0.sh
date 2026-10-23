#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
poll_seconds="${CROSS_SUBJECT_GPU0_POLL_SECONDS:-60}"
lock_file="/tmp/wanzhen_myproject_new_gpu0_cross_subject_autogluon_default.lock"

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

exec 9>"${lock_file}"
echo "$(date '+%F %T') waiting for the GPU0 cross-subject AutoGluon-default lock" >&2
flock 9
echo "$(date '+%F %T') acquired the GPU0 cross-subject AutoGluon-default lock" >&2

gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
while true; do
  active="$({ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits || true; } \
    | awk -F',' -v uuid="${gpu_uuid}" '$1 == uuid {gsub(/ /, "", $2); print $2}' \
    | paste -sd, -)"
  if [[ -z "${active}" ]]; then
    break
  fi
  echo "$(date '+%F %T') waiting for physical GPU0; active compute PIDs=${active}" >&2
  sleep "${poll_seconds}"
done

runner="scripts/run_tridataset_initial_a1_cross_subject_autogluon_default.py"

"${python_bin}" -u "${runner}" \
  check --dataset pamap2 --gpu-id 0 --num-cpus 8 --time-limit 300 --num-bag-folds 2

"${python_bin}" -u "${runner}" \
  check --dataset mhealth --gpu-id 0 --num-cpus 8 --time-limit 300 --num-bag-folds 2

"${python_bin}" -u "${runner}" \
  train-test --dataset pamap2 --gpu-id 0 --num-cpus 8 --time-limit 300 \
  --num-bag-folds 2 --resume

exec "${python_bin}" -u "${runner}" \
  train-test --dataset mhealth --gpu-id 0 --num-cpus 8 --time-limit 300 \
  --num-bag-folds 2 --resume
