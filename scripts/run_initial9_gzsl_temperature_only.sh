#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^(pamap2|mhealth)$ || ! "$2" =~ ^[01]$ ]]; then
  echo "Usage: $0 {pamap2|mhealth} {0|1} [check|train-test|report] [--resume] [--split-indices ...]" >&2
  exit 2
fi
experiment_dataset=$1
experiment_gpu=$2
shift 2
experiment_mode=${1:-train-test}
if [[ $# -gt 0 ]]; then shift; fi
experiment_output=/data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42
experiment_python=/home/wanzhen/miniconda3/envs/zara/bin/python
experiment_script=/home/wanzhen/myproject_new/scripts/run_initial9_gzsl_temperature_only.py

# Validate the real filesystem before making logs/caches/locks.
"$experiment_python" -I -B - "$experiment_output" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).resolve()
if not p.is_relative_to(Path('/data/wanzhen').resolve()):
    raise SystemExit('Output must be under /data/wanzhen')
ancestor = p
while not ancestor.exists():
    ancestor = ancestor.parent
if ancestor.stat().st_dev != Path('/data').stat().st_dev or ancestor.stat().st_dev == Path('/').stat().st_dev:
    raise SystemExit('Output must physically reside on the data filesystem')
p.mkdir(parents=True, exist_ok=True)
PY
mkdir -p "$experiment_output/$experiment_dataset/cache/pycache" \
  "$experiment_output/$experiment_dataset/cache/tmp" "$experiment_output/locks"
export CUDA_VISIBLE_DEVICES="$experiment_gpu" PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
export PYTHONPYCACHEPREFIX="$experiment_output/$experiment_dataset/cache/pycache"
export TMPDIR="$experiment_output/$experiment_dataset/cache/tmp"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

if [[ "$experiment_mode" == train-test ]]; then
  # Serialize these launchers per dataset and GPU; preserve already running jobs.
  exec 8>"$experiment_output/locks/$experiment_dataset.lock"
  flock -n 8 || { echo "A launcher for $experiment_dataset already exists" >&2; exit 1; }
  exec 9>"$experiment_output/locks/gpu$experiment_gpu.lock"
  flock 9
  experiment_uuid=$(nvidia-smi -i "$experiment_gpu" --query-gpu=uuid --format=csv,noheader)
  experiment_idle_count=0
  while (( experiment_idle_count < 2 )); do
    experiment_apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
    if awk -F, -v uuid="$experiment_uuid" '$1 == uuid {found=1} END {exit !found}' <<< "$experiment_apps"; then
      experiment_idle_count=0
      echo "[$(date '+%F %T %Z')] GPU $experiment_gpu is busy; waiting 30s ($experiment_dataset)."
    else
      experiment_idle_count=$((experiment_idle_count + 1))
    fi
    if (( experiment_idle_count < 2 )); then sleep 30; fi
  done
  echo "[$(date '+%F %T %Z')] Starting $experiment_dataset on physical GPU $experiment_gpu."
fi

exec "$experiment_python" -u "$experiment_script" "$experiment_dataset" "$experiment_mode" \
  --gpu-id "$experiment_gpu" --output-root "$experiment_output" "$@"
