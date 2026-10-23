import json
import hashlib

import pytest

from zshar.semantics import completion


def _plan():
    return {
        "attributes": [{
            "name": "cadence",
            "definition": "Rate of periodic motion.",
            "levels": {"Low": "slow", "Medium": "moderate", "High": "fast"},
        }]
    }


def test_completion_calls_kami_compatible_transport(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan()))
    output = tmp_path / "vectors.json"
    calls = []

    def fake_transport(**kwargs):
        calls.append(kwargs)
        return json.dumps({"walking": {"cadence": "Medium"}})

    monkeypatch.setattr(completion, "call_llm", fake_transport)
    monkeypatch.setenv("NEW_API_KEY", "test-only-placeholder")
    completion.main([
        "--dataset_name", "mhealth",
        "--a2_plan_json", str(plan),
        "--activities", "walking",
        "--output_json", str(output),
        "--api-format", "openai",
        "--base-url", "https://www.kamiapi.top/v1",
        "--model", "gpt-5.5",
        "--stream",
    ])

    assert json.loads(output.read_text()) == {"walking": {"cadence": "Medium"}}
    assert len(calls) == 1
    assert calls[0]["api_format"] == "openai"
    assert calls[0]["api_key"] == "test-only-placeholder"
    assert calls[0]["base_url"] == "https://www.kamiapi.top/v1"
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["stream"] is True


def test_completion_dry_run_does_not_require_api_key(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan()))
    output = tmp_path / "vectors.json"

    def unexpected_call(**kwargs):
        raise AssertionError("dry-run must not call the API")

    monkeypatch.setattr(completion, "call_llm", unexpected_call)
    completion.main([
        "--a2_plan_json", str(plan),
        "--activities", "walking",
        "--output_json", str(output),
        "--dry_run",
    ])

    assert output.with_suffix(".prompt.txt").exists()
    assert not output.exists()


def test_completion_regenerates_all_values_when_plan_falls_back(tmp_path, monkeypatch):
    a1 = tmp_path / "a1.json"
    a1.write_text(
        json.dumps({"walking": {"cadence": "Medium"}}, indent=4) + "\n",
        encoding="utf-8",
    )
    a1_hash = hashlib.sha256(a1.read_bytes()).hexdigest()
    fallback_plan = {
        **_plan(),
        "_meta": {
            "binding_decision": "fallback_a1",
            "a1_json": str(a1),
            "a1_sha256": a1_hash,
        },
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(fallback_plan), encoding="utf-8")
    output = tmp_path / "vectors.json"
    report = tmp_path / "report.json"

    calls = []

    def fake_transport(**kwargs):
        calls.append(kwargs)
        return json.dumps({"walking": {"cadence": "High"}})

    monkeypatch.setattr(completion, "call_llm", fake_transport)
    completion.main([
        "--a2_plan_json", str(plan),
        "--activities_json", str(a1),
        "--output_json", str(output),
        "--report_json", str(report),
    ])

    assert json.loads(output.read_text()) == {"walking": {"cadence": "High"}}
    assert len(calls) == 1
    audit = json.loads(report.read_text(encoding="utf-8"))
    assert audit["vector_source"]["mode"] == completion.VECTOR_GENERATION_MODE
    assert audit["vector_source"]["llm_called"] is True
    assert audit["vector_source"]["copied_keep_attributes"] == []
    assert output.with_suffix(".prompt.txt").exists()
    assert output.with_suffix(".raw_response.txt").exists()


def _incremental_plan():
    cadence = {
        "name": "cadence",
        "definition": "Rate of periodic motion.",
        "levels": {"Low": "slow", "Medium": "moderate", "High": "fast"},
    }
    vertical = {
        "name": "vertical_smoothness",
        "definition": "Continuity of gravity-axis motion.",
        "levels": {"Low": "abrupt", "Medium": "mixed", "High": "continuous"},
    }
    horizontal = {
        "name": "horizontal_smoothness",
        "definition": "Continuity of horizontal-plane motion.",
        "levels": {"Low": "abrupt", "Medium": "mixed", "High": "continuous"},
    }
    return {
        "attributes": [cadence, vertical, horizontal],
        "decision_reason": "Walking and jumping need clearer directional continuity.",
        "semantic_redundancy_review": "The child concepts cover distinct directions.",
        "operations": [
            {
                "action": "keep", "inputs": ["cadence"], "outputs": ["cadence"],
                "reason": "Cadence remains useful.", "target_unseen": [],
                "contrast_unseen": [], "seen_anchors": [],
            },
            {
                "action": "split", "inputs": ["smoothness"],
                "outputs": ["vertical_smoothness", "horizontal_smoothness"],
                "reason": "Separate jump impacts from continuous walking trajectories.",
                "target_unseen": ["jumping"], "contrast_unseen": ["walking"],
                "seen_anchors": ["running"],
            },
        ],
        "_meta": {
            "seen": ["running"],
            "unseen": ["walking", "jumping"],
        },
    }


def test_completion_generates_full_matrix_in_one_call_including_keep(tmp_path, monkeypatch):
    a1_vectors = {
        "walking": {"cadence": "Medium", "smoothness": "High"},
        "jumping": {"cadence": "Low", "smoothness": "Low"},
        "running": {"cadence": "High", "smoothness": "Medium"},
    }
    a1 = tmp_path / "a1.json"
    a1.write_text(json.dumps(a1_vectors))
    plan_payload = _incremental_plan()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_payload))
    output = tmp_path / "vectors.json"
    calls = []

    changed_values = {
        "walking": {"cadence": "High", "vertical_smoothness": "High", "horizontal_smoothness": "High"},
        "jumping": {"cadence": "Medium", "vertical_smoothness": "Low", "horizontal_smoothness": "Medium"},
        "running": {"cadence": "High", "vertical_smoothness": "Medium", "horizontal_smoothness": "High"},
    }

    def fake_transport(**kwargs):
        calls.append(kwargs)
        return json.dumps(changed_values)

    monkeypatch.setattr(completion, "call_llm", fake_transport)
    monkeypatch.setenv("NEW_API_KEY", "test-only-placeholder")
    completion.main([
        "--dataset_name", "mhealth",
        "--a2_plan_json", str(plan),
        "--a1-json", str(a1),
        "--activities_json", str(a1),
        "--output_json", str(output),
    ])

    vectors = json.loads(output.read_text())
    assert vectors == changed_values
    assert len(calls) == 1
    assert vectors["walking"]["cadence"] == "High"
    assert vectors["jumping"]["cadence"] == "Medium"
    assert vectors["walking"]["vertical_smoothness"] == "High"
    prompt = calls[0]["messages"][0]["content"]
    assert "Separate jump impacts" in prompt
    assert '"target_unseen": [\n        "jumping"' in prompt
    assert '"walking": {' in prompt and '"cadence": "Medium"' in prompt
    assert "including keep operations" in prompt
    assert "copied deterministically" not in prompt
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["vector_source"]["copied_keep_attributes"] == []
    assert report["vector_source"]["llm_generated_attributes"] == [
        "cadence", "vertical_smoothness", "horizontal_smoothness"
    ]
    completion.validate_saved_generation(
        vector_path=output, report_path=output.with_suffix(".report.json"),
        plan_path=plan, raw_path=output.with_suffix(".raw_response.txt"), a1_path=a1,
    )
    plan.write_text(plan.read_text() + "\n")
    with pytest.raises(ValueError, match="plan hash"):
        completion.validate_saved_generation(
            vector_path=output, report_path=output.with_suffix(".report.json"),
            plan_path=plan, raw_path=output.with_suffix(".raw_response.txt"),
        )


def test_all_keep_plan_still_generates_all_values(tmp_path, monkeypatch):
    a1 = tmp_path / "a1.json"
    a1.write_text(json.dumps({"walking": {"cadence": "Medium"}}))
    plan_payload = {
        **_plan(),
        "decision_reason": "No change needed.",
        "semantic_redundancy_review": "No overlap.",
        "operations": [{
            "action": "keep", "inputs": ["cadence"], "outputs": ["cadence"],
            "reason": "Keep cadence.", "target_unseen": [],
            "contrast_unseen": [], "seen_anchors": [],
        }],
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_payload))
    output = tmp_path / "vectors.json"
    calls = []

    def fake_transport(**kwargs):
        calls.append(kwargs)
        return json.dumps({"walking": {"cadence": "High"}})

    monkeypatch.setattr(completion, "call_llm", fake_transport)
    completion.main([
        "--a2_plan_json", str(plan), "--a1-json", str(a1),
        "--activities_json", str(a1), "--output_json", str(output),
    ])
    assert json.loads(output.read_text()) == {"walking": {"cadence": "High"}}
    assert len(calls) == 1
    audit = json.loads(output.with_suffix(".report.json").read_text())
    assert audit["vector_source"]["llm_called"] is True
    assert audit["vector_source"]["copied_keep_attributes"] == []


def test_changed_only_response_cannot_be_filled_from_a1():
    plan = _incremental_plan()
    partial = {"walking": {"vertical_smoothness": "High", "horizontal_smoothness": "Low"}}
    with pytest.raises(ValueError, match="[Aa]ttribute"):
        completion.assemble_incremental_vectors(
            plan=plan, activities=["walking"],
            a1_vectors={"walking": {"cadence": "Medium", "smoothness": "High"}},
            llm_output=partial,
        )
    with pytest.raises(ValueError, match="complete LLM response"):
        completion.assemble_full_vectors(plan=plan, activities=["walking"], llm_output=None)


@pytest.mark.parametrize('runner', [
    'run_pamap2_a2_vector_completion_kami',
    'run_mhealth_a2_vector_completion_kami',
    'run_wisdm_a2_vector_completion_kami',
    'run_mhealth_all_splits',
])
def test_batch_recovery_requires_full_generation_provenance(tmp_path, runner):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / 'scripts' / f'{runner}.py'
    spec = importlib.util.spec_from_file_location(runner, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {
        'plan': tmp_path / 'plan.json', 'vectors': tmp_path / 'vectors.json',
        'raw': tmp_path / 'raw.txt', 'report': tmp_path / 'report.json',
    }
    paths['plan'].write_text(json.dumps(_plan()))
    paths['raw'].write_text(json.dumps({'walking': {'cadence': 'High'}}))
    a1 = tmp_path / 'a1.json'
    a1_vectors = {'walking': {'cadence': 'Low'}}
    a1.write_text(json.dumps(a1_vectors))
    completion.record_generation_response(paths['raw'], paths['plan'], a1)

    def recover():
        if runner == 'run_mhealth_all_splits':
            return module.recover_vectors_from_raw(paths, paths['plan'], ['walking'], a1)
        return module.recover_raw(paths, ['walking'], ['cadence'], a1_vectors)

    def valid():
        if runner == 'run_mhealth_all_splits':
            return module.valid_vectors(paths, paths['plan'], ['walking'], a1)[0]
        return module.valid_vectors(paths, ['walking'], ['cadence'])[0]

    assert recover()
    assert json.loads(paths['vectors'].read_text()) == {'walking': {'cadence': 'High'}}
    assert valid()
    # A structurally complete legacy matrix still cannot pass as newly generated.
    report = json.loads(paths['report'].read_text())
    report['vector_source']['mode'] = 'incremental_from_a1'
    paths['report'].write_text(json.dumps(report))
    assert not valid()
    completion.response_provenance_path(paths['raw']).unlink()
    assert not recover()


def test_completion_does_not_overwrite_a1_input(tmp_path, monkeypatch):
    plan, a1 = tmp_path / 'plan.json', tmp_path / 'a1.json'
    plan.write_text(json.dumps(_plan()))
    original = json.dumps({'walking': {'cadence': 'Low'}})
    a1.write_text(original)
    monkeypatch.setattr(completion, 'call_llm', lambda **_: pytest.fail('input collision called API'))
    with pytest.raises(ValueError, match='must not overwrite inputs'):
        completion.main([
            '--a2_plan_json', str(plan), '--activities_json', str(a1),
            '--output_json', str(a1),
        ])
    assert a1.read_text() == original
