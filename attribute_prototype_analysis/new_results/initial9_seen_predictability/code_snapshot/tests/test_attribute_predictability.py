import numpy as np
import pandas as pd
import pytest

from zshar.evaluation.attribute_predictability import (
    aggregate_matrices, align_probabilities, audit_partition, score_attribute,
)


def test_eod_uses_probabilities_and_declared_ordinal_spacing():
    # Missing Medium still leaves High/Low at full distance 1, not a compressed scale.
    proba = pd.DataFrame({"High": [.25, .9], "Low": [.75, .1]})
    metrics, sample = score_attribute(["Low", "Medium"], ["Low", "High"], proba)
    np.testing.assert_allclose(sample, [.25, .5])
    assert metrics["eod"] == .375
    assert align_probabilities(proba).columns.tolist() == ["Low", "Medium", "High"]


def test_noncontiguous_probability_columns_are_aligned_by_name():
    proba = pd.DataFrame({"High": [.7], "Low": [.1], "Medium": [.2]})
    metrics, _ = score_attribute(["Low"], ["High"], proba)
    assert metrics["eod"] == pytest.approx(.8)


def test_constant_attribute_is_flagged_and_fixed_level_f1_is_separate():
    metrics, _ = score_attribute(["Low", "Low"], ["Low", "Low"], pd.DataFrame({"Low": [1, 1]}))
    assert metrics["macro_f1"] == 1
    assert metrics["macro_f1_all_levels"] == pytest.approx(1/3)
    assert metrics["validation_state_count"] == 1
    assert metrics["eod"] == 0


def test_float32_ensemble_roundoff_is_repaired_without_hiding_invalid_probabilities():
    table = align_probabilities(pd.DataFrame({"High": [1.0000001192092896], "Low": [0.]}))
    assert table.loc[0, "High"] == 1
    assert table.sum(axis=1).iloc[0] == 1
    with pytest.raises(ValueError):
        align_probabilities(pd.DataFrame({"High": [1.001], "Low": [0.]}))


@pytest.mark.parametrize("proba", [pd.DataFrame({"unknown": [1]}), pd.DataFrame({"Low": [.9]}),
                                   pd.DataFrame({"Low": [np.nan]}), pd.DataFrame({"Low": [-.1], "High": [1.1]})])
def test_bad_probabilities_are_rejected(proba):
    with pytest.raises(ValueError):
        align_probabilities(proba)


def test_outer_validation_is_excluded_from_both_fitting_and_tuning():
    assert audit_partition([1, 2], [3], [1, 2, 3], [4, 5])["outer_overlap"] == 0
    with pytest.raises(ValueError):
        audit_partition([1, 2], [3, 4], [1, 2, 3], [4, 5])
    with pytest.raises(ValueError):
        audit_partition([1], [2], [1, 2, 3], [4])


def test_splits_have_equal_weight_regardless_of_validation_size():
    rows = pd.DataFrame([dict(dataset="d", attribute="a", split_index=1, eod=.1, n=10),
                         dict(dataset="d", attribute="a", split_index=2, eod=.9, n=90)])
    mean, std, count = aggregate_matrices(rows, ["a"], ["d"], "eod")
    assert mean.loc["a", "d"] == .5
    assert std.loc["a", "d"] == pytest.approx(np.std([.1, .9], ddof=1))
    assert count.loc["a", "d"] == 2
    with pytest.raises(ValueError):
        aggregate_matrices(pd.concat([rows, rows]), ["a"], ["d"], "eod")
