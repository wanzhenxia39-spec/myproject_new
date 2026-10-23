#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
src_root="${project_root}/src"

a1_json="${project_root}/configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json"
ontology_json="${project_root}/configs/attributes/initial_attributes.json"
gap_csv="${project_root}/runtime/gap/pamap2/a1_gap_v1_5_seed42.csv"
output_root="${project_root}/runtime/refinement_semantic_v3/pamap2_zzz_gpt_5_6_sol"
log_root="${project_root}/runtime/logs/pamap2_refinement_semantic_v3_zzz_gpt_5_6_sol"

: "${ZZZ_API_KEY:?请先执行 export ZZZ_API_KEY='你的API Key'}"

mkdir -p "${output_root}" "${log_root}"
cd "${project_root}"

mapfile -t split_indices < <(
  "${python_bin}" -c "
import pandas as pd
frame = pd.read_csv('${gap_csv}')
for value in frame['split_index'].astype(int):
    if value != 13:
        print(value)
"
)

echo "需要处理的split数量：${#split_indices[@]}"
echo "split列表：${split_indices[*]}"

for split_index in "${split_indices[@]}"; do
  split_dir="${output_root}/split_${split_index}"
  output_json="${split_dir}/a2_attribute_plan.json"
  log_file="${log_root}/split_${split_index}.log"

  if [[ -s "${output_json}" ]]; then
    echo "[SKIP] split ${split_index} 已存在：${output_json}"
    continue
  fi

  mkdir -p "${split_dir}"

  echo "[START] split ${split_index}"

  env \
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
    --api-format openai \
    --base-url https://api.zhizengzeng.com/v1 \
    --model gpt-5.6-sol \
    --temperature 0 \
    --stream \
    --timeout 600 \
    2>&1 | tee "${log_file}"

  echo "[DONE] split ${split_index}"
done

echo "剩余PAMAP2属性细化任务全部完成。"
