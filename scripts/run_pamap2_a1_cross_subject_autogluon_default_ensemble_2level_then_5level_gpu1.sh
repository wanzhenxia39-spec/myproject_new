#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
runner="${project_root}/scripts/run_pamap2_a1_cross_subject_autogluon_default_ensemble_level_ablation_gpu0.py"
lock_file="/tmp/wanzhen_myproject_new_gpu1_cross_subject_autogluon_default.lock"

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

exec 9>"${lock_file}"
flock -n 9 || {
  echo "$(date '+%F %T') another GPU1 PAMAP2 cross-subject queue holds ${lock_file}" >&2
  exit 1
}

echo "$(date '+%F %T') preflight: PAMAP2 2-level cross-subject on physical GPU1" >&2
"${python_bin}" -u "${runner}" \
  check --level-count 2 --gpu-id 1 --num-cpus 8 --time-limit 300 --num-bag-folds 2

echo "$(date '+%F %T') training/evaluation: PAMAP2 2-level cross-subject on physical GPU1" >&2
"${python_bin}" -u "${runner}" \
  train-test --level-count 2 --gpu-id 1 --num-cpus 8 --time-limit 300 \
  --num-bag-folds 2 --resume

echo "$(date '+%F %T') preflight: PAMAP2 5-level cross-subject on physical GPU1" >&2
"${python_bin}" -u "${runner}" \
  check --level-count 5 --gpu-id 1 --num-cpus 8 --time-limit 300 --num-bag-folds 2

echo "$(date '+%F %T') training/evaluation: PAMAP2 5-level cross-subject on physical GPU1" >&2
exec "${python_bin}" -u "${runner}" \
  train-test --level-count 5 --gpu-id 1 --num-cpus 8 --time-limit 300 \
  --num-bag-folds 2 --resume
