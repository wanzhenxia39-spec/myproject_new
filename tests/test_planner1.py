"""Contracts for semantic-only full-A1 refinement."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from zshar.features.groups import load_groups
from zshar.semantics import planner1


def _concept(name):
    return {
        "name": name,
        "definition": f"Physical meaning of {name}",
        "levels": {"Low": "low degree", "Medium": "medium degree", "High": "high degree"},
    }


def gap_info():
    return {
        "split_index": 13,
        "seen": ["walking", "running"],
        "unseen": ["cycling", "jumping"],
        "semantic_gap": {
            "D_kNN": 0.4,
            "D_kNN_per_unseen": {"cycling": 0.3, "jumping": 0.5},
            "U_NN": 0.1,
            "U_NN_per_unseen": {"cycling": 0.1, "jumping": 0.1},
            "nearest_seen_topk": {
                "cycling": [["running", 0.2]],
                "jumping": [["walking", 0.4]],
            },
            "top_k": 1,
            "effective_k": 1,
            "distance_metric": "mad",
        },
        "a1_sha256": None,
    }


def split_evidence():
    return planner1.build_split_evidence(
        "mhealth",
        gap_info(),
        {
            "walking": {"cadence": "Medium"},
            "running": {"cadence": "High"},
            "cycling": {"cadence": "High"},
            "jumping": {"cadence": "Low"},
        },
        [_concept("cadence")],
    )


def keep_plan():
    return {
        "schema_version": "2.0",
        "dataset": "mhealth",
        "split_index": 13,
        "refinement_needed": False,
        "decision_reason": "The complete A1 matrix needs no change.",
        "attributes": [_concept("cadence")],
        "operations": [{
            "action": "keep",
            "inputs": ["cadence"],
            "outputs": ["cadence"],
            "reason": "Useful transferable repetition rate.",
            "target_unseen": [],
            "contrast_unseen": [],
            "seen_anchors": [],
        }],
        "semantic_redundancy_review": "No overlapping concepts.",
    }


def validation_context():
    return {
        "a1_names": ["cadence"],
        "seen": ["walking", "running"],
        "unseen": ["cycling", "jumping"],
        "dataset": "mhealth",
        "split_index": 13,
        "min_attributes": 1,
        "max_attributes": 15,
    }


def write_gap(path):
    pd.DataFrame([{
        "split_index": 13,
        "seen_classes": "walking|running",
        "unseen_classes": "cycling|jumping",
        "D_kNN": 0.4,
        "U_NN": 0.1,
        "D_kNN_per_unseen_json": json.dumps({"cycling": 0.3, "jumping": 0.5}),
        "U_NN_per_unseen_json": json.dumps({"cycling": 0.1, "jumping": 0.1}),
        "nearest_seen_topk_json": json.dumps({
            "cycling": [["running", 0.2]], "jumping": [["walking", 0.4]],
        }),
        "top_k": 1,
        "effective_k": 1,
        "distance_metric": "mad",
    }]).to_csv(path, index=False)


def test_gap_csv_diagnostics_are_reused_including_u_nn(tmp_path):
    path = tmp_path / "gap.csv"
    write_gap(path)
    resolved = planner1.resolve_split_from_gap_csv(path, [], [], 13)
    assert resolved["semantic_gap"]["D_kNN"] == 0.4
    assert resolved["semantic_gap"]["U_NN"] == 0.1
    assert resolved["semantic_gap"]["U_NN_per_unseen"]["cycling"] == 0.1
    assert resolved["semantic_gap"]["nearest_seen_topk"]["jumping"] == [["walking", 0.4]]


def test_prompt_is_semantic_only_and_targets_close_pairs():
    evidence = split_evidence()
    capabilities = {
        "feature_groups": {"G6": {"name": "Frequency", "physical_capability": "periodicity"}},
        "sensor_inventory": {"hand": {"body_position": "arm", "modalities": ["acc", "gyro"]}},
    }
    messages = planner1.build_messages(evidence, capabilities)
    assert [message["role"] for message in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "zero-shot" in user
    assert "differ in only one or two coordinates" in user
    assert "high D_kNN" in user
    assert "low U_NN" in user
    assert "Sensor modality, body position or motion axis may be used" in user
    assert "primary_groups" not in user
    assert "feature_selectors" not in user
    assert "selector_profiles" not in user
    assert "candidate refinements" not in user
    assert "issue-card" not in user


def test_capabilities_have_no_selector_profiles():
    capabilities = planner1.compact_physical_capabilities(
        "pamap2", ["hand_gyro_x_mean", "hand_CAGH"], load_groups()
    )
    assert set(capabilities["feature_groups"]) == {"G2", "G14"}
    assert all("selector_profiles" not in value for value in capabilities["feature_groups"].values())
    assert capabilities["sensor_inventory"]["hand"]["body_position"] == "hand"


def test_semantic_validation_preserves_keep_and_allows_sensor_scoped_names():
    plan = keep_plan()
    assert planner1.validate_single_stage_response(plan, split_evidence(), validation_context()) is plan

    changed = deepcopy(plan)
    changed["attributes"][0]["definition"] = "Silently changed semantics"
    with pytest.raises(ValueError, match="preserve"):
        planner1.validate_single_stage_response(changed, split_evidence(), validation_context())

    planner1._validate_attribute_name("cadence")
    planner1._validate_attribute_name("hand_motion")
    planner1._validate_attribute_name("vertical_axis_motion")
    with pytest.raises(ValueError, match="snake_case"):
        planner1._validate_attribute_name("hand-motion")


def test_pipeline_calls_llm_once_and_writes_no_feature_mapping(tmp_path, monkeypatch):
    evidence = split_evidence()
    a1 = tmp_path / "a1.json"
    ontology = tmp_path / "ontology.json"
    gap = tmp_path / "gap.csv"
    a1.write_text(json.dumps(evidence["a1_class_prototypes"]))
    ontology.write_text(json.dumps(evidence["a1_ontology"]))
    write_gap(gap)
    calls = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        return json.dumps(keep_plan())

    monkeypatch.setattr(planner1, "call_llm", fake_llm)
    output = tmp_path / "run" / "a2_attribute_plan.json"
    result = planner1.run_split(
        dataset="mhealth",
        a1_path=a1,
        ontology_path=ontology,
        gap_path=gap,
        gap_info=planner1.resolve_split_from_gap_csv(gap, [], [], 13),
        output_path=output,
        feature_names=["hand_acc_x_fft_dom_freq"],
        feature_source="fixture",
        config=load_groups(),
        min_attributes=1,
        api_key="test",
    )
    assert result["status"] == "validated_semantic_a2"
    assert len(calls) == 1
    plan = json.loads(output.read_text())
    assert plan["_meta"]["gap_diagnostics_source"] == "reused_from_gap_csv"
    assert plan["_meta"]["validation_decision"] == "validated_semantic_a2"
    assert not (tmp_path / "run/a2_attribute_plan.feature_mapping.json").exists()


def test_dry_run_writes_prompts_without_calling_llm(tmp_path, monkeypatch):
    evidence = split_evidence()
    a1 = tmp_path / "a1.json"
    ontology = tmp_path / "ontology.json"
    gap = tmp_path / "gap.csv"
    a1.write_text(json.dumps(evidence["a1_class_prototypes"]))
    ontology.write_text(json.dumps(evidence["a1_ontology"]))
    write_gap(gap)
    monkeypatch.setattr(
        planner1, "call_llm",
        lambda **_: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    output = tmp_path / "preview/a2_attribute_plan.json"
    result = planner1.run_split(
        dataset="mhealth", a1_path=a1, ontology_path=ontology, gap_path=gap,
        gap_info=planner1.resolve_split_from_gap_csv(gap, [], [], 13), output_path=output,
        feature_names=["hand_acc_x_fft_dom_freq"], feature_source="fixture",
        config=load_groups(), dry_run=True,
    )
    assert result["status"] == "prompt_only"
    assert not output.exists()
    assert (tmp_path / "preview/a2_attribute_plan.prompt.json").is_file()
    assert (tmp_path / "preview/a2_attribute_plan.prompt.txt").is_file()
