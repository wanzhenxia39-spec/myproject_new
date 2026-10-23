#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/home/wanzhen/myproject_new"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
run_root="/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs"
gpu_id=0
poll_seconds=60
required_idle_checks=2
lock_file="/tmp/wanzhen_mhealth_a2_gzsl_autogluon_default_gpu0.lock"
pamap_run="${run_root}/pamap2/pamap2_gzsl_autogluon_default_ensemble_a2_semantic_v3_independent_calibrated_grid_cal3_t300_random20_eval20_seen20_seed42_gpu0"
pamap_summary="${pamap_run}/summary.csv"
pamap_report="${pamap_run}/gzsl_report.csv"

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another MHEALTH GZSL AutoGluon-default GPU-0 waiter/run holds ${lock_file}."
  exit 1
fi

pamap_running() {
  pgrep -f '[s]cripts/run_pamap2_a2_gzsl_autogluon_default_ensemble.py train-test' \
    >/dev/null
}

pamap_complete() {
  "${python_bin}" - "${pamap_summary}" "${pamap_report}" <<'PY'
import csv
from pathlib import Path
import sys

summary = Path(sys.argv[1])
report = Path(sys.argv[2])
if not summary.is_file() or not report.is_file():
    raise SystemExit(1)
with summary.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
indices = {int(row["split_index"]) for row in rows}
raise SystemExit(0 if len(rows) == 20 and len(indices) == 20 else 1)
PY
}

gpu_has_compute_processes() {
  local gpu_uuid query
  if ! gpu_uuid="$(
    nvidia-smi --id="${gpu_id}" --query-gpu=uuid --format=csv,noheader 2>&1
  )"; then
    echo "$(date '+%F %T') Unable to query GPU ${gpu_id}: ${gpu_uuid}"
    return 0
  fi
  gpu_uuid="${gpu_uuid//[[:space:]]/}"
  if ! query="$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>&1
  )"; then
    echo "$(date '+%F %T') Unable to query GPU processes: ${query}"
    return 0
  fi
  grep -Fq "${gpu_uuid}," <<<"${query}"
}

echo "Waiting for the matched PAMAP2 GZSL run to complete successfully..."
while pamap_running; do
  completed="$({ find "${pamap_run}" -mindepth 2 -maxdepth 2 -name metrics.json 2>/dev/null || true; } | wc -l)"
  echo "$(date '+%F %T') PAMAP2 progress: ${completed}/20; checking again in ${poll_seconds}s."
  sleep "${poll_seconds}"
done

if ! pamap_complete; then
  echo "$(date '+%F %T') PAMAP2 process ended without a valid 20-split final report; MHEALTH will not start."
  exit 1
fi

echo "$(date '+%F %T') PAMAP2 completed 20/20 splits; waiting for physical GPU ${gpu_id}."
idle_checks=0
while (( idle_checks < required_idle_checks )); do
  if gpu_has_compute_processes; then
    idle_checks=0
    echo "$(date '+%F %T') GPU ${gpu_id} is busy; checking again in ${poll_seconds}s."
  else
    ((idle_checks += 1))
    echo "$(date '+%F %T') GPU ${gpu_id} idle confirmation ${idle_checks}/${required_idle_checks}."
  fi
  if (( idle_checks < required_idle_checks )); then
    sleep "${poll_seconds}"
  fi
done

echo "$(date '+%F %T') Starting matched MHEALTH GZSL AutoGluon default ensemble."
cd "${project_root}"
export PYTHONPATH="${project_root}/src"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
exec "${python_bin}" -u \
  scripts/run_mhealth_a2_gzsl_autogluon_default_ensemble.py \
  train-test \
  --gpu-id 0 \
  --num-cpus 8 \
  --time-limit 300 \
  --run-root "${run_root}" \
  --resume
