#!/usr/bin/env bash
# DSADS full A2 vectors, fixed seed-42 ZSL, AutoGluon default ensembles.
# Optional first argument: check | train-test | summarize (default: train-test).
# Remaining options are forwarded to the shared Python launcher.
# --after-uschad waits for successful completion of USC-HAD on GPU 0.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
if [[ "${project_root}" != /data/wanzhen/* ]]; then
  echo "Run the migrated DSADS launcher under /data/wanzhen; resolved project: ${project_root}" >&2
  exit 2
fi
python_bin="${V4_PYTHON_BIN:-/home/wanzhen/miniconda3/envs/zara/bin/python}"
mode="train-test"
if (($#)) && [[ "$1" != -* ]]; then
  mode="$1"
  shift
fi
after_uschad=false
forward_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--after-uschad" ]]; then
    after_uschad=true
  else
    forward_args+=("${arg}")
  fi
done

cd "${project_root}"
if [[ "${mode}" == "train-test" ]]; then
  # AutoGluon and its dependencies also create temporary files and caches.
  export TMPDIR="${project_root}/runtime/tmp/dsads_v4"
  export JOBLIB_TEMP_FOLDER="${TMPDIR}/joblib"
  export XDG_CACHE_HOME="${project_root}/runtime/cache"
  export TORCH_HOME="${XDG_CACHE_HOME}/torch"
  mkdir -p "${TMPDIR}" "${JOBLIB_TEMP_FOLDER}" "${TORCH_HOME}"
fi
if "${after_uschad}"; then
  if [[ "${mode}" != "train-test" ]]; then
    echo "--after-uschad is only supported for train-test" >&2
    exit 2
  fi
  "${python_bin}" -u "${script_dir}/wait_for_uschad_v4.py"
fi
exec "${python_bin}" -u "${script_dir}/run_dsads_v4_autogluon_default_ensemble.py" \
  "${mode}" --gpu-id 0 --time-limit 300 --num-cpus 8 --resume \
  --attribute-vectors-root "${project_root}/configs/attrIbutes_v4/dsads" \
  --run-root "${project_root}/runtime/zshar_runs" "${forward_args[@]}"
