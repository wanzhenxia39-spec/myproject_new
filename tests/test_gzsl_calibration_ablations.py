from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zshar.evaluation.calibration import calibrated_scores, tune_calibration


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_pamap2_a2_gzsl_autogluon_soft_oof_calibration_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("pamap2_gzsl_calibration_ablation", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_temperature_only_has_no_seen_penalty_and_requires_gamma_zero():
    scores = pd.DataFrame({"seen": [2.0], "unseen": [1.0]})
    adjusted = calibrated_scores(
        scores,
        ["seen"],
        gamma=0.0,
        temperature=2.0,
        bias_correction="temperature_only",
    )
    assert np.isclose(float(adjusted.sum(axis=1).iloc[0]), 1.0)
    assert adjusted.idxmax(axis=1).iloc[0] == "seen"
    with pytest.raises(ValueError, match="gamma must be 0.0"):
        calibrated_scores(
            scores,
            ["seen"],
            gamma=0.1,
            temperature=2.0,
            bias_correction="temperature_only",
        )


def test_temperature_only_tunes_nll_while_fixing_gamma_zero():
    scores = pd.DataFrame({"a": [3.0, 0.0], "b": [0.0, 3.0]})
    folds = [
        {
            "scores": scores,
            "truth": ["a", "b"],
            "pseudo_seen": ["a"],
            "pseudo_unseen": ["b"],
        }
    ]
    result = tune_calibration(
        folds,
        mode="grid",
        gamma_values=[-1.0, 0.0, 1.0],
        temperature_grid=[0.5, 1.0, 2.0],
        tpe_trials=5,
        tau_min=0.1,
        tau_max=10.0,
        seed=42,
        bias_correction="temperature_only",
    )
    assert result["best_gamma"] == 0.0
    assert result["best_temperature"] == 0.5
    assert result["selection_metric"] == "mean_validation_negative_log_likelihood"


def test_ablation_configs_are_strictly_distinct_and_use_old_gzsl_manifests():
    assert str(runner.base.SPLITS).startswith("/home/wanzhen/myproject/split/gzsl_20/")
    assert str(runner.base.SAMPLES).startswith("/home/wanzhen/myproject/split/gzsl_20/")
    expected = {
        "temperature_only": ("grid", "temperature_only", 0.0, 1.0),
        "seen_penalty_only": ("grid", "seen_penalty_only", 0.0, 1.0),
        "no_calibration": ("fixed", "none", 0.0, 1.0),
    }
    for variant, values in expected.items():
        args = runner.build_parser().parse_args(
            ["check", "--variant", variant, "--gpu-id", "0"]
        )
        config = runner.experiment_config(args, [1])
        assert (
            config.calibration_mode,
            config.gzsl_bias_correction,
            config.gamma,
            config.temperature,
        ) == values
        if variant in {"temperature_only", "no_calibration"}:
            assert config.gamma_grid is None


def test_no_calibration_is_explicit_raw_score_identity():
    scores = pd.DataFrame({"seen": [2.0], "unseen": [1.0]})
    adjusted = calibrated_scores(
        scores,
        ["seen"],
        gamma=0.0,
        temperature=1.0,
        bias_correction="none",
    )
    pd.testing.assert_frame_equal(adjusted, scores)
