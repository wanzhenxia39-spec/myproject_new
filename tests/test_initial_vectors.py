import json

import pytest

from zshar.semantics import initial_vectors
from zshar.semantics.completion import validate_llm_output
from zshar.semantics.ontology import load_ontology

EXPECTED_ATTRIBUTES = (
    "motion_intensity", "cadence", "smoothness", "impulsiveness",
    "posture_inclination", "posture_variability", "rotational_motion",
    "translational_motion", "motion_regularity",
)


@pytest.mark.parametrize("dataset", initial_vectors.available_initial_datasets())
def test_dataset_prompt_uses_nine_semantic_attributes_without_early_group_binding(dataset):
    context = initial_vectors.load_initial_context(dataset)
    messages = initial_vectors.build_prompts(context)
    payload = json.loads(messages[1]["content"])
    attrs = payload["semantic_attribute_ontology"]["attributes"]
    assert [a["name"] for a in attrs] == list(EXPECTED_ATTRIBUTES)
    assert all(set(a) == {"name", "definition", "levels"} for a in attrs)
    assert "supporting_feature_groups" not in payload
    assert "sensors_retained_by_preprocessing" not in payload
    assert "dataset_metadata" not in payload
    assert payload["activities"] == context["activities"]
    assert payload["dataset_specific_comparison_focus"]
    assert "structured reasoning process internally" in messages[0]["content"]
    assert "most easily confused activities" in messages[0]["content"]
    assert "old\nten-attribute schema" in messages[0]["content"]
    assert "no feature groups" in messages[0]["content"]
    assert "Do not infer or design any feature-group correspondence" in messages[0]["content"]


def test_dataset_specific_prompt_context_is_required():
    context = initial_vectors.load_initial_context("pamap2")
    with pytest.raises(ValueError, match="prompt context"):
        initial_vectors.build_prompts(context, prompt_context={})
    with pytest.raises(ValueError, match="comparison_focus"):
        initial_vectors.build_prompts(context, prompt_context={"pamap2": {}})


def test_output_validation_rejects_old_schema_and_invalid_levels():
    names = list(EXPECTED_ATTRIBUTES)
    good = {"walking": {name: "Medium" for name in names}}
    assert validate_llm_output(good, ["walking"], names) == good
    bad = {"walking": {**good["walking"], "body_coordination": "High"}}
    with pytest.raises(ValueError, match="attribute mismatch"):
        validate_llm_output(bad, ["walking"], names)
    good["walking"]["cadence"] = "Fast"
    with pytest.raises(ValueError, match="Invalid level"):
        validate_llm_output(good, ["walking"], names)


def test_generation_and_dry_run_without_network(tmp_path, monkeypatch):
    def unexpected_call(**kwargs):
        pytest.fail("dry-run and response-file modes must not call the API")
    monkeypatch.setattr(initial_vectors, "call_llm", unexpected_call)
    output = tmp_path / "vectors.json"
    args = ["--dataset", "pamap2", "--output", str(output)]
    initial_vectors.main([*args, "--dry-run"])
    assert not output.exists()
    assert output.with_suffix(".prompt.json").exists()
    readable = output.with_suffix(".prompt.txt")
    assert readable.exists()
    readable_text = readable.read_text()
    assert "SYSTEM" in readable_text and "USER" in readable_text
    assert '"semantic_attribute_ontology": {' in readable_text
    assert '\\nGenerate one class-level' not in readable_text
    activities = initial_vectors.load_initial_context("pamap2")["activities"]
    vectors = {a: {name: "Low" for name in EXPECTED_ATTRIBUTES} for a in activities}
    response = tmp_path / "response.txt"
    response.write_text("```json\n" + json.dumps(vectors) + "\n```")
    initial_vectors.main([*args, "--response-json", str(response)])
    assert json.loads(output.read_text()) == vectors
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["num_attributes"] == 9
    assert report["duplicate_full_vectors"]
    assert report["policy"]["filtering"] is False
    assert report["readable_prompt_path"].endswith(".prompt.txt")


def test_live_path_passes_prompt_to_transport(tmp_path, monkeypatch):
    context = initial_vectors.load_initial_context("uschad")
    vectors = {activity: {name: "Medium" for name in EXPECTED_ATTRIBUTES}
               for activity in context["activities"]}
    calls = []
    def fake_transport(**kwargs):
        calls.append(kwargs)
        return json.dumps(vectors)
    monkeypatch.setattr(initial_vectors, "call_llm", fake_transport)
    monkeypatch.setenv("LLM_API_KEY", "test-only-placeholder")
    output = tmp_path / "vectors.json"
    initial_vectors.main(["--dataset", "uschad", "--output", str(output)])
    assert len(calls) == 1
    assert len(calls[0]["messages"]) == 2
    assert calls[0]["stream"] is True
    assert json.loads(output.read_text()) == vectors


def test_kami_key_environment_aliases(monkeypatch):
    for name in ("LLM_API_KEY", "KAMI_API_KEY", "NEW_API_KEY", "OPENAI_API_KEY", "ZZZ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEW_API_KEY", "kami-test-placeholder")
    assert initial_vectors.resolve_api_key("openai") == "kami-test-placeholder"
    assert initial_vectors.resolve_api_key("gemini") == ""


def test_initial_ontology_contains_semantics_only():
    ontology = load_ontology()
    assert set(ontology) == {"version", "attributes"}
    assert [attr["name"] for attr in ontology["attributes"]] == list(EXPECTED_ATTRIBUTES)
    assert all(set(attr) == {"name", "definition", "levels"}
               for attr in ontology["attributes"])
