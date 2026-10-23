#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
runner="${project_root}/scripts/run_pamap2_a1_cross_subject_autogluon_default_ensemble_level_ablation_gpu0.py"
poll_seconds="${PAMAP2_CROSS_SUBJECT_GPU0_POLL_SECONDS:-60}"
idle_polls_required="${PAMAP2_CROSS_SUBJECT_GPU0_IDLE_POLLS:-2}"
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
idle_polls=0
while (( idle_polls < idle_polls_required )); do
  active_gpu_pids="$({ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits || true; } \
    | awk -F',' -v uuid="${gpu_uuid}" '$1 == uuid {gsub(/ /, "", $2); print $2}' \
    | paste -sd, -)"
  active_gzsl_jobs="$(pgrep -af \
    '[r]un_pamap2_a1_gzsl_autogluon_default_ensemble_(2level_then_5level_gpu0\.sh|level_ablation_gpu0\.py)' \
    || true)"
  if [[ -z "${active_gpu_pids}" && -z "${active_gzsl_jobs}" ]]; then
    ((idle_polls += 1))
    echo "$(date '+%F %T') GPU0 idle check ${idle_polls}/${idle_polls_required}" >&2
  else
    idle_polls=0
    echo "$(date '+%F %T') waiting for GPU0; compute_pids=${active_gpu_pids:-none}" >&2
    if [[ -n "${active_gzsl_jobs}" ]]; then
      echo "${active_gzsl_jobs}" >&2
    fi
  fi
  if (( idle_polls < idle_polls_required )); then
    sleep "${poll_seconds}"
  fi
done

echo "$(date '+%F %T') preflight: PAMAP2 2-level cross-subject" >&2
"${python_bin}" -u "${runner}" \
  check --level-count 2 --gpu-id 0 --num-cpus 8 --time-limit 300 --num-bag-folds 2

echo "$(date '+%F %T') training/evaluation: PAMAP2 2-level cross-subject" >&2
"${python_bin}" -u "${runner}" \
  train-test --level-count 2 --gpu-id 0 --num-cpus 8 --time-limit 300 \
  --num-bag-folds 2 --resume

echo "$(date '+%F %T') preflight: PAMAP2 5-level cross-subject" >&2
"${python_bin}" -u "${runner}" \
  check --level-count 5 --gpu-id 0 --num-cpus 8 --time-limit 300 --num-bag-folds 2

echo "$(date '+%F %T') training/evaluation: PAMAP2 5-level cross-subject" >&2
exec "${python_bin}" -u "${runner}" \
  train-test --level-count 5 --gpu-id 0 --num-cpus 8 --time-limit 300 \
  --num-bag-folds 2 --resume
