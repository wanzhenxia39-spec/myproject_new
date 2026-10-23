#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_attributes_v4_pipeline.sh" --dataset wisdm --gpu-id 1 "$@"
