#!/usr/bin/env python3
"""Run PAMAP2 A1-v1.5 CatBoost-order AutoGluon soft-OOF forward chains."""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CORE_SCRIPT = (
    PROJECT / "scripts/run_pamap2_a1_autogluon_soft_oof_catboost_orders.py"
)
SPEC = importlib.util.spec_from_file_location("pamap2_a1_soft_oof_forward_core", CORE_SCRIPT)
core = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(core)

core.__doc__ = __doc__
core.VARIANTS = ("forward",)


if __name__ == "__main__":
    raise SystemExit(core.main())
