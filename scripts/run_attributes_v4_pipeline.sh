#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${V4_PYTHON_BIN:-/home/wanzhen/miniconda3/envs/zara/bin/python}"
dataset=""
gpu_id=""
stage="all"
vector_root=""
model="gpt-6.1-sol"
base_url="https://xxapi.info/v1"
temperature="0.7"
time_limit="300"
num_cpus="8"

usage() {
  cat <<'EOF'
Usage: run_attributes_v4_pipeline.sh --dataset uschad|wisdm [options]
  --gpu-id N                    Physical GPU; USC-HAD defaults to 0, WISDM to 1.
  --stage all|generate|train     Default all: generate, compare GAP, then train/test.
  --attribute-vectors-root DIR  Default configs/attrIbutes_v4/<dataset>.
  --model MODEL                 Default gpt-6.1-sol.
  --base-url URL                Default https://xxapi.info/v1.
  --temperature VALUE           Default 0.7 for both LLM stages.
  --time-limit SECONDS          Per-attribute AutoGluon time limit; default 300.
  --num-cpus N                  Per-training-process CPU count; default 8.
  -h, --help                    Show help.

NEW_API_KEY is used only for pending LLM requests. Validated stages resume automatically.
The training stage uses all 20 seed-42 ZSL splits and exports per-attribute test accuracy.
EOF
}

while (($#)); do
  case "$1" in
    --dataset) dataset="${2:?--dataset requires a value}"; shift 2 ;;
    --gpu-id) gpu_id="${2:?--gpu-id requires a value}"; shift 2 ;;
    --stage) stage="${2:?--stage requires a value}"; shift 2 ;;
    --attribute-vectors-root) vector_root="${2:?--attribute-vectors-root requires a value}"; shift 2 ;;
    --model) model="${2:?--model requires a value}"; shift 2 ;;
    --base-url) base_url="${2:?--base-url requires a value}"; shift 2 ;;
    --temperature) temperature="${2:?--temperature requires a value}"; shift 2 ;;
    --time-limit) time_limit="${2:?--time-limit requires a value}"; shift 2 ;;
    --num-cpus) num_cpus="${2:?--num-cpus requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${dataset}" in
  uschad) gpu_id="${gpu_id:-0}" ;;
  wisdm) gpu_id="${gpu_id:-1}" ;;
  *) echo "--dataset must be uschad or wisdm" >&2; exit 2 ;;
esac
case "${stage}" in
  all|generate|train) ;;
  *) echo "--stage must be all, generate, or train" >&2; exit 2 ;;
esac
if ! [[ "${gpu_id}" =~ ^[0-9]+$ && "${time_limit}" =~ ^[1-9][0-9]*$ && "${num_cpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU must be nonnegative; time limit and CPU count must be positive integers" >&2
  exit 2
fi
vector_root="${vector_root:-${project_root}/configs/attrIbutes_v4/${dataset}}"
cd "${project_root}"
export PYTHONUNBUFFERED=1

if [[ "${stage}" == "all" || "${stage}" == "generate" ]]; then
  echo "[1/2] ${dataset}: latest semantic refinement, full A2 matrices, and GAP comparison"
  "${python_bin}" -u "${script_dir}/run_${dataset}_refinement_xxapi.py" \
    --root "${vector_root}" --model "${model}" --base-url "${base_url}" \
    --temperature "${temperature}" --min-attributes 9 --max-attributes 15
fi

if [[ "${stage}" == "all" || "${stage}" == "train" ]]; then
  echo "[2/2] ${dataset}: independent AutoGluon default ensemble ZSL on physical GPU ${gpu_id}"
  "${python_bin}" -u "${script_dir}/run_${dataset}_v4_autogluon_default_ensemble.py" train-test \
    --attribute-vectors-root "${vector_root}" --gpu-id "${gpu_id}" \
    --time-limit "${time_limit}" --num-cpus "${num_cpus}" \
    --resume --wait-for-vectors --poll-seconds 60 \
    --wait-for-gpu --gpu-poll-seconds 60 --gpu-max-used-mib 1024 --gpu-idle-checks 2
fi

echo "Complete: dataset=${dataset}, stage=${stage}"
