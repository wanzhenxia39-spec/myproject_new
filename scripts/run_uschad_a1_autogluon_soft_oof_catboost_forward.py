#!/usr/bin/env python3
"""Run USC-HAD A1-v1.5 CatBoost-order AutoGluon soft-OOF forward chains.

For every seed-42 ZSL split, independent CatBoost classifiers are fitted only
on seen-class sensor samples. Their validation ranking determines the
split-specific forward order for the nine-dimensional A1 attribute chain.
AutoGluon then trains a bagged soft-OOF probability chain and evaluates the
fixed 30-window-per-unseen-activity sample manifest.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CORE_SCRIPT = (
    PROJECT / "scripts/run_pamap2_a1_autogluon_soft_oof_catboost_orders.py"
)
SPEC = importlib.util.spec_from_file_location("a1_soft_oof_catboost_core", CORE_SCRIPT)
core = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(core)

core.__doc__ = __doc__
core.DATASET_KEY = "uschad"
core.DATASET_LABEL = "USC-HAD"
core.EVAL_SAMPLES_PER_ACTIVITY = 30
core.DEFAULT_GPU_ID = 1
core.A1 = (
    PROJECT / "configs/attributes/activity_attribute_vectors_initial_uschad_v1_5.json"
)
core.SPLITS = (
    PROJECT / "split/zero_shot_split/uschad_unseen4_random20_splits_seed42.json"
)
core.SAMPLES = (
    PROJECT
    / "split/zero_shot_split/uschad_unseen4_random20_eval30_samples_seed42.json"
)
core.PROTOCOL_ROOT = PROJECT / "runtime/uschad_a1_catboost_seen_only_order_seed42"
core.ORDER_WORK = core.PROTOCOL_ROOT / "order_preparation"
core.ORDER_FILE = (
    PROJECT
    / "configs/attribute_orders/uschad/initial_a1_v1_5_catboost_seen_only_seed42.json"
)
core.RUN_NAME_PREFIX = "uschad_zsl_autogluon_soft_oof_a1_v1_5_catboost_seen_only"
core.VARIANTS = ("forward",)


if __name__ == "__main__":
    raise SystemExit(core.main())
