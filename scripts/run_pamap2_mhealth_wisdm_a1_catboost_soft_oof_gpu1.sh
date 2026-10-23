#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wanzhen/myproject_new
export PYTHONPATH=/home/wanzhen/myproject_new/src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

# Prevent duplicate A1 three-dataset queues from using physical GPU 1.
lock_file=/tmp/wanzhen_myproject_new_gpu1_catboost_soft_oof.lock
exec 9>"${lock_file}"
echo "$(date '+%F %T') waiting for the GPU 1 CatBoost soft-OOF queue lock" >&2
flock 9
echo "$(date '+%F %T') acquired the GPU 1 CatBoost soft-OOF queue lock" >&2

# Do not start until every compute process has released physical GPU 1.
gpu_uuid="$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
while true; do
  active="$({ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits || true; } \
    | awk -F',' -v uuid="${gpu_uuid}" '$1 == uuid {gsub(/ /, "", $2); print $2}' \
    | paste -sd, -)"
  [[ -z "${active}" ]] && break
  echo "$(date '+%F %T') waiting for physical GPU 1; active compute PIDs=${active}" >&2
  sleep 60
done

exec /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_tridataset_catboost_soft_oof.py \
  train-test --profile a1 --gpu-id 1 --num-cpus 8
