import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discriminability import (
    analyze_prototypes, encode_prototypes, load_prototypes,
)


def test_hand_calculated_pair_counts_and_full_vector_collisions():
    # Six unordered pairs. Only a/b collide on the complete vector.
    table = encode_prototypes({
        "a": {"x": "Low", "y": "Low"},
        "b": {"x": "Low", "y": "Low"},
        "c": {"x": "Medium", "y": "Low"},
        "d": {"x": "High", "y": "High"},
    })
    result = analyze_prototypes(table)
    attrs = result["attribute_metrics"].set_index("attribute")
    assert attrs.loc["x", "different_pairs"] == 5
    assert attrs.loc["y", "different_pairs"] == 3
    assert attrs.loc["x", "coverage"] == pytest.approx(5 / 6)
    assert attrs.loc["y", "coverage"] == 0.5
    assert result["summary"]["exact_apcr"] == pytest.approx(1 / 6)
    assert result["summary"]["mean_single_attribute_collision_rate"] == pytest.approx(1 / 3)
    pairs = result["pair_metrics"]
    assert len(pairs) == 6
    assert pairs.loc[pairs.exact_collision == 1, ["activity_i", "activity_j"]].values.tolist() == [["a", "b"]]
    distance = result["distance_matrix"]
    assert distance.loc["a", "c"] == 0.25
    assert distance.loc["a", "d"] == 1
    np.testing.assert_array_equal(distance, distance.T)
    np.testing.assert_array_equal(np.diag(distance), np.zeros(4))


def test_single_coordinate_collision_is_not_complete_prototype_collision():
    result = analyze_prototypes(encode_prototypes({
        "a": {"x": 0, "y": 2}, "b": {"x": 0, "y": 0}, "c": {"x": 2, "y": 0},
    }))
    assert result["summary"]["exact_apcr"] == 0
    assert result["summary"]["mean_single_attribute_collision_rate"] == pytest.approx(1 / 3)


def test_fixed_scale_even_with_absent_high_and_dimension_invariance():
    table = encode_prototypes({"a": {"x": "Low"}, "b": {"x": "Medium"}})
    assert analyze_prototypes(table)["distance_matrix"].iloc[0, 1] == 0.5
    repeated = encode_prototypes({"a": {"x": 0, "y": 0}, "b": {"x": 1, "y": 1}})
    assert analyze_prototypes(repeated)["distance_matrix"].iloc[0, 1] == 0.5
    binary = encode_prototypes({"a": {"x": 0}, "b": {"x": 1}}, level_map={}, value_range=(0, 1))
    assert analyze_prototypes(binary)["distance_matrix"].iloc[0, 1] == 1


def test_collision_extremes_and_coverage_complement():
    result = analyze_prototypes(encode_prototypes({"a": {"x": "Medium"}, "b": {"x": "Medium"}}))
    assert result["summary"]["exact_apcr"] == 1
    assert result["summary"]["mean_attribute_coverage"] == 0
    for dataset in (result, analyze_prototypes(encode_prototypes({"a": {"x": 0}, "b": {"x": 2}}))):
        attrs = dataset["attribute_metrics"]
        np.testing.assert_allclose(attrs.coverage + attrs.single_attribute_collision_rate, 1)


@pytest.mark.parametrize("vectors", [
    {}, {"a": {"x": 0}}, {"a": {"x": 0}, "b": {"y": 0}},
    {"a": {"x": "Unknown"}, "b": {"x": 0}},
    {"a": {"x": float("nan")}, "b": {"x": 0}},
    {"a": {"x": 3}, "b": {"x": 0}},
    {"a": {"x": True}, "b": {"x": 0}},
])
def test_invalid_inputs_fail_without_silent_imputation(vectors):
    with pytest.raises(ValueError):
        encode_prototypes(vectors)


def test_csv_and_json_have_identical_results_and_align_by_attribute_name(tmp_path):
    json_path = tmp_path / "vectors.json"
    json_path.write_text(json.dumps({"a": {"x": "Low", "y": "High"},
                                     "b": {"y": "Medium", "x": "High"}}))
    csv_path = tmp_path / "vectors.csv"
    csv_path.write_text("activity,x,y\na,Low,High\nb,High,Medium\n")
    np.testing.assert_array_equal(load_prototypes(json_path).values, load_prototypes(csv_path).values)
    json_path.write_text('{"a":{"x":0},"a":{"x":2}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_prototypes(json_path)
    csv_path.write_text("activity,x,x\na,0,1\nb,1,2\n")
    with pytest.raises(ValueError, match="unique headers"):
        load_prototypes(csv_path)


def test_random_matrices_match_independent_combinatorial_formulas():
    # Coverage = 1 - sum_l binom(n_l,2)/binom(C,2); full APCR counts duplicate rows.
    rng = np.random.default_rng(42)
    for classes in (2, 12, 19):
        raw = rng.integers(0, 3, size=(classes, 9))
        vectors = {f"c{i}": {f"a{k}": int(value) for k, value in enumerate(row)}
                   for i, row in enumerate(raw)}
        result = analyze_prototypes(encode_prototypes(vectors))
        total_pairs = classes * (classes - 1) // 2
        same_counts = []
        for column in raw.T:
            counts = np.bincount(column, minlength=3)
            same_counts.append(sum(int(n) * (int(n) - 1) // 2 for n in counts))
        np.testing.assert_allclose(result["attribute_metrics"].coverage,
                                   1 - np.asarray(same_counts) / total_pairs)
        _, counts = np.unique(raw, axis=0, return_counts=True)
        expected = sum(int(n) * (int(n) - 1) // 2 for n in counts) / total_pairs
        assert result["summary"]["exact_apcr"] == expected
