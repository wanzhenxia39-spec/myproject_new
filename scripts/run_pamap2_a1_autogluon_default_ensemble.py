#!/usr/bin/env python3
"""Run PAMAP2 new-version initial-A1 ZSL AutoGluon default ensemble."""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT / "scripts/run_pamap2_initial9_autogluon_default_ensemble.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "pamap2_initial_a1_autogluon_default_base", BASE_SCRIPT
)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

# Explicitly use the requested seed-42 manifests from myproject_new.
base.SPLITS = (
    PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
)
base.SAMPLES = (
    PROJECT
    / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
)
base.RUN_ROOT = PROJECT / "runtime/zshar_runs"
base.RUN_NAME = (
    "pamap2_zsl_autogluon_default_ensemble_initial_a1_v1_5_"
    "independent_t300_random20_eval50_seed42"
)


if __name__ == "__main__":
    raise SystemExit(base.main())
