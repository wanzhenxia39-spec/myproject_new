import csv
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attribute_discrimination_coverage import compute_attribute_coverage, build_coverage_matrix, read_vectors


def test_hand_calculated_coverage_and_counts():
    vectors = {"a": {"x": "Low", "y": "yes"}, "b": {"x": "Low", "y": "no"},
               "c": {"x": "Medium", "y": "yes"}, "d": {"x": "High", "y": "no"}}
    rows = {row["attribute"]: row for row in compute_attribute_coverage(vectors)}
    assert rows["x"]["total_pairs"] == 6
    assert rows["x"]["different_pairs"] == 5
    assert rows["x"]["coverage"] == pytest.approx(5 / 6)
    assert rows["y"]["coverage"] == pytest.approx(4 / 6)
    for row in rows.values():
        assert row["coverage"] + row["single_attribute_collision_rate"] == pytest.approx(1)


def test_counter_formula_matches_explicit_pair_enumeration():
    vectors = {f"c{i}": {"x": i % 3, "binary": i % 2, "constant": "same"} for i in range(19)}
    pairs = list(combinations(vectors.values(), 2))
    for row in compute_attribute_coverage(vectors):
        expected = sum(left[row["attribute"]] != right[row["attribute"]] for left, right in pairs)
        assert row["different_pairs"] == expected
        assert row["coverage"] == expected / len(pairs)


def test_matrix_aligns_attribute_names_and_supports_arbitrary_dimensions():
    matrix = build_coverage_matrix({
        "ds1": {"a": {"x": 0, "y": 1}, "b": {"x": 0, "y": 0}},
        "ds2": {"a": {"y": "same", "x": "one"}, "b": {"y": "same", "x": "two"}},
    })
    assert matrix == {"attributes": ["x", "y"], "datasets": ["ds1", "ds2"], "values": [[0, 1], [1, 0]]}
    with pytest.raises(ValueError, match="Attributes differ"):
        build_coverage_matrix({"a": {"c1": {"x": 0}, "c2": {"x": 1}},
                               "b": {"c1": {"y": 0}, "c2": {"y": 1}}})


@pytest.mark.parametrize("vectors", [
    {}, {"a": {"x": 0}}, {"a": {"x": 0}, "b": {"y": 0}},
    {"a": {"x": None}, "b": {"x": 0}}, {"a": {"x": "NaN"}, "b": {"x": 0}},
])
def test_invalid_prototypes_are_rejected(vectors):
    with pytest.raises(ValueError):
        compute_attribute_coverage(vectors)


def test_csv_and_json_levels_are_equivalent(tmp_path):
    source = tmp_path / "vectors.csv"
    source.write_text("activity,x,y\na,0,same\nb,1,different\nc,1,same\n")
    json_vectors = {"a": {"x": 0, "y": "same"}, "b": {"x": 1, "y": "different"},
                    "c": {"x": 1, "y": "same"}}
    assert compute_attribute_coverage(read_vectors(source)) == compute_attribute_coverage(json_vectors)
    source = tmp_path / "duplicate.json"
    source.write_text('{"a":{"x":0},"a":{"x":1}}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_vectors(source)


def test_batch_cli_is_standalone_and_separates_methods_with_different_attributes(tmp_path):
    entries = []
    for method, attributes in [("method_one", ["x", "y"]), ("method_two", ["z"])]:
        for dataset in ("ds1", "ds2"):
            source = tmp_path / f"{method}_{dataset}.json"
            source.write_text(json.dumps({"a": {name: "low" for name in attributes},
                                          "b": {name: "high" for name in attributes}}))
            entries.append({"method": method, "dataset": dataset, "vectors": source.name})
    manifest = tmp_path / "methods.json"
    manifest.write_text(json.dumps(entries))
    output = tmp_path / "results"
    script = Path(__file__).resolve().parents[1] / "attribute_discrimination_coverage.py"
    subprocess.run([sys.executable, "-B", "-I", "-S", str(script), "--manifest", str(manifest),
                    "--output-dir", str(output)], check=True, capture_output=True, text=True)
    for method, expected_rows in [("method_one", 2), ("method_two", 1)]:
        with (output / method / "attribute_coverage_matrix.csv").open() as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ["attribute", "ds1", "ds2"]
            rows = list(reader)
            assert len(rows) == expected_rows
            assert all(float(row[ds]) == 1 for row in rows for ds in ("ds1", "ds2"))
