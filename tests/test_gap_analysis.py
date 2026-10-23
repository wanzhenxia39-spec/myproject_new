import json
from dataclasses import replace

import pandas as pd
import pytest

from zshar.cli import build_parser
from zshar.datasets import available_datasets, load_dataset_spec
from zshar.semantics.gap import (
    compute_gap, d_knn, attribute_distance, normalized_ordinal_prototype_distance,
)
from zshar.semantics.gap_analysis import planner_gap_row, run_gap_analysis
from zshar.semantics.ontology import initial_attribute_names
from zshar.semantics.planner import resolve_split_from_gap_csv
from zshar.splits import ActivitySplit


def test_dknn_averages_nearest_seen_then_unseen():
    # u1 -> 0, 0.5; u2 -> 0.5, 1: average of per-class means is 0.5.
    vectors = {"s1": [0, 0], "s2": [0.5, 0.5], "u1": [0, 0], "u2": [1, 1]}
    score, per_class = d_knn(vectors, ["s1", "s2"], ["u1", "u2"], [1, 1], k=2)
    assert per_class == {"u1": 0.25, "u2": 0.75}
    assert score == 0.5
    assert d_knn(vectors, ["s1", "s2"], ["u1", "u2"], [1, 1], k=20) == (score, per_class)


@pytest.mark.parametrize("metric", ["normalized_ordinal", "mad", "euclidean"])
def test_adapter_preserves_core_results_and_exports_neighbors(metric):
    split = ActivitySplit(7, (0, 1), (2, 3), ("s1", "s2"), ("u1", "u2"))
    vectors = {"s1": {"x": "Low", "y": "Low"},
               "s2": {"x": "Medium", "y": "High"},
               "u1": {"x": "Low", "y": "Medium"},
               "u2": {"x": "High", "y": "High"}}
    original = compute_gap(split, vectors, k=2, metric=metric)
    adapted = planner_gap_row(split, vectors, k=2, metric=metric)
    for key, value in original.items():
        assert adapted[key] == value
    neighbors = json.loads(adapted["nearest_seen_topk_json"])
    for activity, entries in neighbors.items():
        assert len(entries) == 2
        assert {item[0] for item in entries} == {"s1", "s2"}
        assert sum(item[1] for item in entries) / 2 == pytest.approx(original["D_kNN_per_unseen"][activity])
    with pytest.raises(ValueError, match="disjoint"):
        planner_gap_row(replace(split, unseen_activities=("s1",)), vectors)


@pytest.mark.parametrize("dataset", available_datasets())
def test_latest_a1_csv_can_be_loaded_by_planner(tmp_path, dataset):
    spec = load_dataset_spec(dataset)
    attrs = initial_attribute_names()
    levels = ["Low", "Medium", "High"]
    vectors = {activity: {name: levels[(i + j) % 3] for j, name in enumerate(attrs)}
               for i, activity in enumerate(spec.activity_names)}
    source = tmp_path / "a1.json"
    source.write_text(json.dumps(vectors))
    destination = tmp_path / "gap.csv"
    run_gap_analysis(dataset=dataset, a1_json=source, output=destination)
    frame = pd.read_csv(destination)
    assert len(frame) == 20
    for _, row in frame.iterrows():
        parsed = resolve_split_from_gap_csv(str(destination), None, None, int(row.split_index))
        assert parsed["seen"] and parsed["unseen"]
        assert set(parsed["seen"]) | set(parsed["unseen"]) == set(vectors)
        assert set(parsed["D_kNN_per_unseen"]) == set(parsed["unseen"])
        assert set(parsed["nearest_seen_topk"]) == set(parsed["unseen"])
        assert parsed["D_kNN"] == pytest.approx(sum(parsed["D_kNN_per_unseen"].values()) / len(parsed["unseen"]))
    old_hash = frame.iloc[0].a1_sha256
    # Same pathname, newly generated contents: rerunning must read the new values.
    source.write_text(json.dumps({a: {name: "Low" for name in attrs} for a in vectors}))
    run_gap_analysis(dataset=dataset, a1_json=source, output=destination)
    current = pd.read_csv(destination)
    assert (current.D_kNN == 0).all()
    assert current.iloc[0].a1_sha256 != old_hash
    metadata = json.loads(destination.with_suffix(".meta.json").read_text())
    assert metadata["a1_json"] == str(source)
    assert metadata["a1_sha256"] == current.iloc[0].a1_sha256
    assert metadata["attributes"] == attrs


def test_invalid_a1_or_manifest_is_not_silently_replaced(tmp_path):
    with pytest.raises(FileNotFoundError, match="A1 activity vectors not found"):
        run_gap_analysis(dataset="pamap2", a1_json=tmp_path / "missing.json")
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"walking": {"body_coordination": "Low"}}))
    with pytest.raises(ValueError, match="Activity key mismatch"):
        run_gap_analysis(dataset="pamap2", a1_json=source)
    source.write_text(json.dumps({a: {name: "Low" for name in initial_attribute_names()}
                                 for a in load_dataset_spec("pamap2").activity_names}))
    with pytest.raises(FileNotFoundError, match="Split manifest not found"):
        run_gap_analysis(dataset="pamap2", a1_json=source, split_manifest=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="top-k"):
        run_gap_analysis(dataset="pamap2", a1_json=source, top_k=0)


def test_zshar_gap_accepts_current_vector_override(tmp_path):
    args = build_parser().parse_args([
        "gap", "--dataset", "pamap2", "--a1-json", str(tmp_path / "a1.json"),
        "--output", str(tmp_path / "gap.csv"),
    ])
    assert args.a1_json == str(tmp_path / "a1.json")
    assert args.top_k == 2


def test_normalized_ordinal_formula_fixed_span_and_dimension_invariance():
    distance = normalized_ordinal_prototype_distance
    assert distance([0, 1, 2], [2, 1, 1]) == pytest.approx(0.5)
    assert distance([0], [1]) == 0.5  # fixed span, even if High is absent
    assert distance([0, 0], [2, 2]) == 1.0
    assert distance([0, 1], [0, 1]) == 0.0
    # Repeating all dimensions must not reward a larger A2 attribute set.
    assert distance([0, 1] * 6, [2, 1] * 6) == distance([0, 1], [2, 1])
    with pytest.raises(ValueError, match="equal weights"):
        attribute_distance([0, 0.5], [1, 1], [0.1, 0.9])
    for left, right in [([], []), ([0], [0, 1]), ([3], [0]), ([float('nan')], [0])]:
        with pytest.raises(ValueError):
            distance(left, right)


def test_normalized_gap_has_hand_computed_metrics_and_matches_uniform_mad():
    split = ActivitySplit(1, (0, 1), (2, 3), ("s1", "s2"), ("u1", "u2"))
    vectors = {
        "s1": {"x": "Low", "y": "Low"},
        "s2": {"x": "High", "y": "High"},
        "u1": {"x": "Medium", "y": "Low"},
        "u2": {"x": "High", "y": "Medium"},
    }
    result = compute_gap(split, vectors)
    assert result["D_kNN"] == 0.5
    assert result["U_NN"] == 0.5
    assert result["U_Conf"] == 0.5
    assert result["JointGap_DkNN_UConf"] == 0.5
    assert result == compute_gap(split, vectors, metric="mad")
    assert planner_gap_row(split, vectors)["distance_metric"] == "normalized_ordinal"


def test_comparison_validates_variable_a2_and_exports_diagnostics(tmp_path):
    from zshar.semantics.gap_comparison import compare_refinement_gap
    spec = load_dataset_spec("pamap2")
    from zshar.splits import default_manifest_paths
    split_path, _ = default_manifest_paths(spec, scenario="zsl", seed=42)
    manifest = json.loads(split_path.read_text())
    manifest["splits"] = manifest["splits"][:2]
    source = tmp_path / "splits.json"
    source.write_text(json.dumps(manifest))
    root = tmp_path / "a2"
    for i, split in enumerate(manifest["splits"]):
        directory = root / f"split_{split['split_index']}"
        directory.mkdir(parents=True)
        vectors = {a: {f"attr_{j}": "Low" for j in range(i + 2)} for a in spec.activity_names}
        (directory / "activity_attribute_vectors_a2.json").write_text(json.dumps(vectors))
    output = tmp_path / "reports"
    result = compare_refinement_gap(dataset="pamap2", a2_vectors_root=root,
                                    split_manifest=source, output_dir=output)
    frame = pd.read_csv(result)
    assert frame["A2_attribute_count"].tolist() == [2, 3]
    assert frame["A2_D_kNN"].tolist() == [0, 0]
    assert frame["A2_U_NN"].tolist() == [0, 0]
    assert frame["A2_U_Conf"].tolist() == [1, 1]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["maximum_absolute_difference_from_uniform_mad"] < 1e-14
    assert len(summary["a2_inputs"]) == 2
    # Missing A2 values must fail before producing a new report.
    bad = root / f"split_{manifest['splits'][0]['split_index']}" / "activity_attribute_vectors_a2.json"
    payload = json.loads(bad.read_text())
    payload[spec.activity_names[0]].pop("attr_0")
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        compare_refinement_gap(dataset="pamap2", a2_vectors_root=root,
                               split_manifest=source, output_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
