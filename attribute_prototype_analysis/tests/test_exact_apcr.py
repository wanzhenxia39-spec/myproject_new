import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_apcr import compute_exact_apcr, result_output, read_vectors


def test_full_vector_equality_and_unordered_pairs():
    result = compute_exact_apcr({
        "a": {"x": "Low", "y": "High"},
        "b": {"y": "High", "x": "Low"},
        "c": {"x": "Low", "y": "Medium"},
        "d": {"x": "High", "y": "High"},
    })
    assert result["total_pairs"] == 6
    assert result["exact_collision_pairs"] == 1
    assert result["exact_apcr"] == pytest.approx(1 / 6)
    assert result["collision_pairs"] == [{"activity_i": "a", "activity_j": "b"}]


def test_group_of_three_counts_all_three_pairs():
    result = compute_exact_apcr({name: {"x": "same"} for name in ("a", "b", "c")})
    assert result["total_pairs"] == result["exact_collision_pairs"] == 3
    assert result["exact_apcr"] == 1


def test_partial_coordinate_equality_is_not_a_collision():
    result = compute_exact_apcr({"a": {"x": 0, "y": 0}, "b": {"x": 0, "y": 1}})
    assert result["exact_apcr"] == 0


def test_numeric_equality_is_exact_without_float_rounding():
    result = compute_exact_apcr({"a": {"x": "1.00000000000000001"},
                                 "b": {"x": "1.00000000000000002"}})
    assert result["exact_apcr"] == 0


@pytest.mark.parametrize("vectors", [
    {}, {"a": {"x": 1}}, {"a": {"x": 0}, "b": {"y": 0}},
    {"a": {"x": None}, "b": {"x": 0}},
    {"a": {"x": True}, "b": {"x": 0}},
    {"a": {"x": "NaN"}, "b": {"x": 0}},
    {"a": {"x": ""}, "b": {"x": 0}},
])
def test_invalid_inputs_are_rejected(vectors):
    with pytest.raises(ValueError):
        compute_exact_apcr(vectors)


def test_cli_is_standalone_without_site_packages_or_other_code(tmp_path):
    # -I excludes the script directory; -S excludes site packages.
    standalone = Path(__file__).resolve().parents[1] / "exact_apcr.py"
    source = tmp_path / "vectors.json"
    source.write_text(json.dumps({"a": {"x": "Low"}, "b": {"x": "Low"}, "c": {"x": "High"}}))
    output = tmp_path / "results"
    result = subprocess.run([sys.executable, "-B", "-I", "-S", str(standalone),
                             "--vectors", str(source), "--dataset", "toy", "--method", "example",
                             "--output-dir", str(output)], check=True, text=True, capture_output=True)
    assert "1/3" in result.stdout
    summary = json.loads((output / "example/toy/summary.json").read_text())
    assert summary["exact_apcr"] == pytest.approx(1 / 3)
    with (output / "method_summary.csv").open() as stream:
        assert float(next(csv.DictReader(stream))["macro_mean_exact_apcr"]) == pytest.approx(1 / 3)


def test_csv_json_equivalence_and_duplicate_key_rejection(tmp_path):
    source = tmp_path / "vectors.csv"
    source.write_text("activity,x,y\na,Low,2\nb,Low,2\nc,High,0\n")
    csv_result = compute_exact_apcr(read_vectors(source))
    json_result = compute_exact_apcr({"a": {"x": "Low", "y": 2}, "b": {"x": "Low", "y": 2},
                                      "c": {"x": "High", "y": 0}})
    assert csv_result == json_result
    source = tmp_path / "bad.json"
    source.write_text('{"a":{"x":0},"a":{"x":1}}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_vectors(source)


def test_data_disk_output_is_rejected_for_this_analysis():
    with pytest.raises(ValueError, match="Output must be under /home/wanzhen/attribute_prototype_analysis/results"):
        result_output("/data/wanzhen/experiments/forbidden_test_output")
