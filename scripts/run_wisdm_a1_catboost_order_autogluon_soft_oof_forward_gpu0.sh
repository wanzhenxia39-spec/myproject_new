#!/usr/bin/env bash
set -euo pipefail

exec /home/wanzhen/myproject_new/scripts/run_wisdm_a1_catboost_order_autogluon_soft_oof_forward.sh \
  --gpu-id 0 \
  "$@"
