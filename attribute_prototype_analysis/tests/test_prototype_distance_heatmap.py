import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_distance_heatmap import (
    encode_prototypes, pairwise_ordinal_distances, pairwise_prototype_distances,
    normalized_attribute_weights, read_attribute_weights, result_output,
)


def test_distance_matches_fixed_span_closed_form():
    table = encode_prototypes({"a": {"x": "Low", "y": "Low"},
                               "b": {"x": "Medium", "y": "Low"},
                               "c": {"x": "High", "y": "High"}})
    matrix = pairwise_ordinal_distances(table)
    np.testing.assert_allclose(matrix, [[0, 0.25, 1], [0.25, 0, 0.75], [1, 0.75, 0]])


def test_standalone_cli_needs_no_sibling_modules(tmp_path):
    source = tmp_path / "prototypes.json"
    source.write_text(json.dumps({"a": {"x": "Low"}, "b": {"x": "Medium"}}))
    output = tmp_path / "heatmap"
    script = Path(__file__).resolve().parents[1] / "prototype_distance_heatmap.py"
    # Isolated mode omits the code directory from sys.path, retaining installed plotting packages.
    subprocess.run([sys.executable, "-B", "-I", str(script), "--dataset", "toy", "--vectors", str(source),
                    "--output-dir", str(output), "--formats", "svg", "--dpi", "100"],
                   check=True, capture_output=True, text=True)
    matrix = pd.read_csv(output / "distance_matrix.csv", index_col=0)
    assert matrix.loc["a", "b"] == 0.5
    metadata = json.loads((output / "distance_run_metadata.json").read_text())
    assert metadata["color_limits"] == [0, 1]
    assert metadata["total_pairs"] == 1
    assert metadata["metric"] == "ordinal_probability"
    assert metadata["weighting"]["policy"] == "uniform"
    assert "<image" not in (output / "prototype_distance_heatmap.svg").read_text()


def test_data_disk_destination_is_rejected():
    with pytest.raises(ValueError, match="Output must be under /home/wanzhen"):
        result_output("/data/wanzhen/experiments/distance_heatmap")


def test_ordinal_probability_reduces_to_l1_for_point_mass_prototypes():
    table = encode_prototypes({"a": {"x": 0, "y": 0}, "b": {"x": 1, "y": 2}})
    expected = [[0, 0.75], [0.75, 0]]
    for metric in ("ordinal_probability", "ordinal_l1", "manhattan"):
        np.testing.assert_allclose(pairwise_prototype_distances(table, metric=metric), expected)


def test_weights_rms_and_hamming_have_hand_calculated_values():
    table = encode_prototypes({"a": {"x": 0, "y": 0}, "b": {"x": 1, "y": 2}})
    weights = {"x": 3.0, "y": 1.0}
    assert pairwise_prototype_distances(table, attribute_weights=weights).iloc[0, 1] == 0.625
    assert pairwise_prototype_distances(table, metric="euclidean", attribute_weights=weights).iloc[0, 1] == pytest.approx(np.sqrt(0.4375))
    assert pairwise_prototype_distances(table, metric="hamming", attribute_weights=weights).iloc[0, 1] == 1.0


def test_cosine_loses_absolute_ordinal_magnitude_and_handles_zero_vectors():
    table = encode_prototypes({"zero": {"x": 0, "y": 0},
                               "medium": {"x": 1, "y": 1}, "high": {"x": 2, "y": 2}})
    distance = pairwise_prototype_distances(table, metric="cosine")
    assert distance.loc["medium", "high"] == pytest.approx(0, abs=1e-15)
    assert distance.loc["zero", "medium"] == 1
    assert distance.loc["zero", "zero"] == 0
    assert pairwise_ordinal_distances(table).loc["medium", "high"] == 0.5


@pytest.mark.parametrize("weights", [{"x": 1}, {"x": -1, "y": 1}, {"x": float("nan"), "y": 1}])
def test_invalid_or_incomplete_weights_are_rejected(weights):
    table = encode_prototypes({"a": {"x": 0, "y": 0}, "b": {"x": 1, "y": 2}})
    with pytest.raises(ValueError):
        normalized_attribute_weights(table, weights)


def test_saved_validation_accuracy_report_is_supported(tmp_path):
    table = encode_prototypes({"a": {"x": 0, "y": 0}, "b": {"x": 1, "y": 2}})
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"per_attribute": {"x": {"accuracy": 0}, "y": {"accuracy": 0.8}}}))
    weights, metadata = read_attribute_weights(path, table)
    assert metadata["policy"] == "validation_accuracy"
    resolved = normalized_attribute_weights(table, weights)
    assert resolved[0] > 0  # Same EPS floor as the final activity matcher.
    assert resolved.sum() == pytest.approx(1)


@pytest.mark.parametrize("metric", ["ordinal_probability", "manhattan", "euclidean", "cosine"])
def test_matches_project_activity_matching_implementation(metric):
    # Verify against the actual matcher without loading models or sample data.
    project_src = Path("/data/wanzhen/projects/myproject_new/src")
    if not project_src.is_dir():
        pytest.skip("Original project source unavailable")
    sys.path.insert(0, str(project_src))
    from zshar.semantics.prototypes import match_prototype
    vectors = {"a": {"x": "Low", "y": "Medium"}, "b": {"x": "Medium", "y": "High"},
               "c": {"x": "High", "y": "Low"}}
    table = encode_prototypes(vectors)
    weights = {"x": 0.7, "y": 0.9}
    matrix = pairwise_prototype_distances(table, metric=metric, attribute_weights=weights)
    for i, left in enumerate(table.activities):
        for j, right in enumerate(table.activities):
            actual = match_prototype(
                predicted_values=vectors[left], prototype_values=vectors[right], attribute_cols=table.attributes,
                metric=metric, attribute_weights=weights, ordinal_values={"Low":0, "Medium":0.5, "High":1},
                probability_distributions={key: {value: 1.0} for key, value in vectors[left].items()},
            )
            assert matrix.iloc[i, j] == pytest.approx(actual.distance, abs=1e-9)
