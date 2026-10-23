#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
src_root="${project_root}/src"

a1_json="${project_root}/configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json"
ontology_json="${project_root}/configs/attributes/initial_attributes.json"
gap_csv="${project_root}/runtime/gap/pamap2/a1_gap_v1_5_seed42.csv"
# Semantic-only v3 plans must not be mixed with the earlier bound-plan artifacts.
output_root="${project_root}/runtime/refinement_semantic_v3/pamap2_gpt_5_6_sol_11_15"
log_root="${project_root}/runtime/logs/pamap2_refinement_semantic_v3_gpt_5_6_sol_11_15"

: "${KAMI_API_KEY:?Please export KAMI_API_KEY for https://www.kamiapi.top/v1}"

mkdir -p "${output_root}" "${log_root}"
cd "${project_root}"

mapfile -t split_indices < <(
  "${python_bin}" -c "
import pandas as pd
frame = pd.read_csv('${gap_csv}')
for value in frame['split_index'].astype(int):
    print(value)
"
)

echo "Provider: Kami (https://www.kamiapi.top/v1)"
echo "Model: gpt-5.6-sol"
echo "Split count: ${#split_indices[@]}"
echo "Splits: ${split_indices[*]}"

failed_splits=()
attempted_splits=()
max_attempts=3

for split_index in "${split_indices[@]}"; do
  split_dir="${output_root}/split_${split_index}"
  output_json="${split_dir}/a2_attribute_plan.json"
  log_file="${log_root}/split_${split_index}.log"

  if [[ -s "${output_json}" ]]; then
    echo "[SKIP VALIDATED] split ${split_index}"
    continue
  fi

  attempted_splits+=("${split_index}")
  mkdir -p "${split_dir}"
  echo "[START] split ${split_index}"

  split_succeeded=false
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    echo "[ATTEMPT ${attempt}/${max_attempts}] split ${split_index}" | tee -a "${log_file}"
    if env \
        -u HTTP_PROXY -u HTTPS_PROXY \
        -u http_proxy -u https_proxy \
        -u ALL_PROXY -u all_proxy \
        PYTHONPATH="${src_root}" \
      "${python_bin}" -u -m zshar.semantics.planner1 \
        --dataset-name pamap2 \
        --a1-json "${a1_json}" \
        --dknn-csv "${gap_csv}" \
        --a1-ontology-json "${ontology_json}" \
        --split-index "${split_index}" \
        --output-json "${output_json}" \
        --min-attributes 11 \
        --max-attributes 15 \
        --api-format openai \
        --base-url https://www.kamiapi.top/v1 \
        --model gpt-5.6-sol \
        --temperature 0 \
        --stream \
        --timeout 600 \
        2>&1 | tee -a "${log_file}"; then
      split_succeeded=true
      break
    fi
    echo "[ATTEMPT FAILED] split ${split_index}, attempt ${attempt}/${max_attempts}" | tee -a "${log_file}"
  done

  if [[ "${split_succeeded}" == true ]]; then
    echo "[DONE] split ${split_index}"
  else
    failed_splits+=("${split_index}")
    echo "[FAILED] split ${split_index} after ${max_attempts} attempts"
  fi
done

echo "Attempted incomplete splits: ${attempted_splits[*]:-none}"
if ((${#failed_splits[@]})); then
  echo "Still failed: ${failed_splits[*]}"
  exit 1
fi

echo "All previously incomplete PAMAP2 refinement tasks completed successfully."
