#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs"
poll_seconds="${PAMAP2_GZSL_ABLATION_GPU0_POLL_SECONDS:-60}"
lock_file="/tmp/wanzhen_myproject_new_gpu0_pamap2_gzsl_calibration_ablations.lock"
runner="scripts/run_pamap2_a2_gzsl_autogluon_soft_oof_calibration_ablation.py"
variants=(no_calibration temperature_only seen_penalty_only)

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

exec 9>"${lock_file}"
echo "$(date '+%F %T') waiting for the GPU0 PAMAP2 GZSL ablation lock" >&2
flock 9
echo "$(date '+%F %T') acquired the GPU0 PAMAP2 GZSL ablation lock" >&2

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

for variant in "${variants[@]}"; do
  "${python_bin}" -u "${runner}" check \
    --variant "${variant}" --gpu-id 0 --num-cpus 8 --time-limit 300 \
    --num-bag-folds 2 --run-root "${run_root}"
done

for variant in "${variants[@]}"; do
  echo "$(date '+%F %T') starting PAMAP2 GZSL ablation: ${variant}" >&2
  "${python_bin}" -u "${runner}" train-test \
    --variant "${variant}" --gpu-id 0 --num-cpus 8 --time-limit 300 \
    --num-bag-folds 2 --run-root "${run_root}" --resume
done

echo "$(date '+%F %T') completed all PAMAP2 GZSL calibration ablations" >&2
