#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
gpu_id="${GPU_ID:-1}"
stage="all"
split_indices=""

usage() {
  cat <<'EOF'
Usage:
  run_mhealth_semantic_v3_full_pipeline_gpu1.sh [options]

Options:
  --stage all|refine|vectors|train  Run all stages or one stage (default: all).
  --gpu-id N                       Physical training GPU (default: 1).
  --split-indices A,B,C            Optional split subset; default: all 20 splits.
  -h, --help                       Show this help.

KAMI_API_KEY is needed only while refinement or vector generation has pending work.
The refinement launcher refreshes the seed-42 A1 GAP CSV when its A1 hash is stale.
Completed, locally validated LLM artifacts are skipped automatically.
EOF
}

while (($#)); do
  case "$1" in
    --stage)
      stage="${2:?--stage requires a value}"
      shift 2
      ;;
    --gpu-id)
      gpu_id="${2:?--gpu-id requires a value}"
      shift 2
      ;;
    --split-indices)
      split_indices="${2:?--split-indices requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${stage}" in
  all|refine|vectors|train) ;;
  *)
    echo "--stage must be all, refine, vectors, or train" >&2
    exit 2
    ;;
esac

if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "--gpu-id must be a nonnegative integer" >&2
  exit 2
fi

cd "${project_root}"
export PYTHONPATH="${project_root}/src"

selection_args=()
if [[ -n "${split_indices}" ]]; then
  selection_args=(--split-indices "${split_indices}")
fi

if [[ "${stage}" == "all" || "${stage}" == "refine" ]]; then
  echo "[STAGE 1/3] MHEALTH semantic attribute refinement"
  "${python_bin}" -u scripts/run_mhealth_refinement_kami.py \
    "${selection_args[@]}"
fi

if [[ "${stage}" == "all" || "${stage}" == "vectors" ]]; then
  echo "[STAGE 2/3] MHEALTH refined attribute-vector completion"
  "${python_bin}" -u scripts/run_mhealth_a2_vector_completion_kami.py \
    "${selection_args[@]}"
fi

if [[ "${stage}" == "all" || "${stage}" == "train" ]]; then
  echo "[STAGE 3/3] MHEALTH seen-only CatBoost order + AutoGluon soft-OOF ZSL"
  "${python_bin}" -u scripts/run_mhealth_a2_autogluon_soft_oof_catboost_forward.py \
    train-test \
    --gpu-id "${gpu_id}" \
    --resume \
    "${selection_args[@]}"
fi
