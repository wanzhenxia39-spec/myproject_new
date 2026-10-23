#!/usr/bin/env bash
# Usage: bash scripts/run_pamap2_mhealth_v4_nonchain.sh pamap2|mhealth [mode] [options]
set -Eeuo pipefail

project_root="/data/wanzhen/projects/myproject_new"
output_root="/data/wanzhen/experiments/pamap2_mhealth_v4_nonchain_seed42"
python_bin="/home/wanzhen/miniconda3/envs/zara/bin/python"
dataset="${1:?Pass pamap2 or mhealth}"
shift
case "$dataset" in
  pamap2|mhealth) ;;
  *) echo "Dataset must be pamap2 or mhealth" >&2; exit 2 ;;
esac

# Set Python's own bytecode/temp/cache locations before the interpreter starts.
runtime_root="${output_root}/${dataset}/runtime"
export TMPDIR="${runtime_root}/tmp" TMP="${runtime_root}/tmp" TEMP="${runtime_root}/tmp"
export PYTHONPYCACHEPREFIX="${runtime_root}/pycache" XDG_CACHE_HOME="${runtime_root}/cache"
# Python validates mount identity and every writable experiment path before use.
# Do not create directories in the shell, where symlinks could bypass that check.
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd /data/wanzhen
exec "$python_bin" -u "${project_root}/scripts/run_v4_nonchain_backend_benchmark.py" "$dataset" "$@"
