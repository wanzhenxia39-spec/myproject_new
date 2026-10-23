#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
expected_backend="${project_root}/src/zshar/models/autogluon_soft_oof_backend.py"
poll_seconds="${SOFT_OOF_GPU0_POLL_SECONDS:-60}"

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

actual_backend="$("${python_bin}" -c \
  'from pathlib import Path; import inspect; from zshar.models.autogluon_soft_oof_backend import AutoGluonSoftOOFModelManager; print(Path(inspect.getfile(AutoGluonSoftOOFModelManager)).resolve())')"
if [[ "${actual_backend}" != "${expected_backend}" ]]; then
  echo "Refusing to run: unexpected soft-OOF backend: ${actual_backend}" >&2
  exit 1
fi
echo "Verified soft-OOF backend: ${actual_backend}"

common_args=(
  --gpu-id 0
  --time-limit 300
  --presets medium_quality
  --num-cpus 8
  --num-bag-folds 2
  --reuse-orders-only
)

echo "Preflight 1/3: PAMAP2 semantic-v3 A2 + cached CatBoost seen-only order"
"${python_bin}" -u scripts/run_pamap2_a2_autogluon_soft_oof_catboost_forward.py \
  check "${common_args[@]}"
echo "Preflight 2/3: MHEALTH semantic-v3 A2 + cached CatBoost seen-only order"
"${python_bin}" -u scripts/run_mhealth_a2_autogluon_soft_oof_catboost_forward.py \
  check "${common_args[@]}"
echo "Preflight 3/3: WISDM semantic-v3 A2 + cached CatBoost seen-only order"
"${python_bin}" -u scripts/run_wisdm_a2_autogluon_soft_oof_catboost_forward.py \
  check "${common_args[@]}"

gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
while true; do
  active="$({ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits || true; } \
    | awk -F',' -v uuid="${gpu_uuid}" '$1 == uuid {gsub(/ /, "", $2); print $2}' \
    | paste -sd, -)"
  if [[ -z "${active}" ]]; then
    break
  fi
  echo "$(date '+%F %T') waiting for physical GPU 0; active compute PIDs=${active}" >&2
  sleep "${poll_seconds}"
done

echo "Stage 1/3: PAMAP2"
"${python_bin}" -u scripts/run_pamap2_a2_autogluon_soft_oof_catboost_forward.py \
  train-test "${common_args[@]}" \
  --run-name-prefix \
  pamap2_zsl_autogluon_soft_oof_a2_semantic_v3_catboost_seen_only_tridataset_gpu0 \
  --resume

echo "Stage 2/3: MHEALTH"
"${python_bin}" -u scripts/run_mhealth_a2_autogluon_soft_oof_catboost_forward.py \
  train-test "${common_args[@]}" \
  --run-name-prefix \
  mhealth_zsl_autogluon_soft_oof_a2_semantic_v3_catboost_seen_only_tridataset_gpu0 \
  --resume

echo "Stage 3/3: WISDM"
"${python_bin}" -u scripts/run_wisdm_a2_autogluon_soft_oof_catboost_forward.py \
  train-test "${common_args[@]}" \
  --run-name-prefix \
  wisdm_zsl_autogluon_soft_oof_a2_semantic_v3_catboost_seen_only_tridataset_gpu0 \
  --resume

echo "All three GPU0 soft-OOF experiments completed."
