#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the complete class-level A2 matrix in one LLM request.

Every final attribute is reassigned, including keep operations and fallback plans.
A1 vectors are optional semantic context; no A1 values are copied into the output.

Input:
  A1 vectors + semantic A2 plan with operation lineage

Output:
  {
    "<activity name>": {
      "<A2 attribute name>": "Low|Medium|High"
    }
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from zshar.llm.transport import call_llm, resolve_api_key as resolve_transport_api_key


VALID_LEVELS = {"Low", "Medium", "High"}
VECTOR_GENERATION_MODE = "full_a2_generation_v1"


def parse_activity_list(text: str) -> List[str]:
    """Parse comma-separated or pipe-separated activity names."""
    if not text:
        return []
    sep = "|" if "|" in text else ","
    return [x.strip() for x in text.split(sep) if x.strip()]


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(data: Any, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(text: str, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def extract_json_object(text: str) -> Any:
    """
    Extract a JSON object from the LLM response.

    The prompt requests JSON-only output, but this helper tolerates accidental
    markdown fences or extra text around the object.
    """
    clean = text.strip()

    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def resolve_api_key(api_format: str, base_url: str = "") -> str:
    """Resolve the key using the target endpoint's provider-specific priority."""
    return resolve_transport_api_key(api_format, base_url)


def get_attribute_names(plan: Mapping[str, Any]) -> List[str]:
    attrs = plan.get("attributes")
    if not isinstance(attrs, list) or not attrs:
        raise ValueError("A2 plan must contain a non-empty 'attributes' list.")

    names: List[str] = []
    for i, attr in enumerate(attrs):
        if not isinstance(attr, Mapping):
            raise ValueError(f"A2 plan attribute at index {i} must be an object.")
        name = attr.get("name")
        if not name:
            raise ValueError(f"A2 plan attribute at index {i} is missing 'name'.")
        names.append(str(name))

    if len(names) != len(set(names)):
        duplicates = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"A2 plan has duplicate attribute names: {duplicates}")
    return names


def semantic_attributes(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Strip planner metadata and retain only the semantics needed for value assignment."""
    return [
        {key: attr[key] for key in ("name", "definition", "levels")}
        for attr in plan["attributes"]
    ]


def operation_generation_spec(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate operation lineage; request every final A2 column from the LLM."""
    final_names = get_attribute_names(plan)
    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations:
        # Compatibility for old schema-only plans: generate every final attribute.
        return {
            "mode": VECTOR_GENERATION_MODE,
            "kept_attributes": [],
            "generated_attributes": final_names,
            "dropped_attributes": [],
            "operations": [],
        }

    kept: List[str] = []
    dropped: List[str] = []
    output_counts: Dict[str, int] = {}
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise ValueError(f"Operation {index} must be an object")
        action = operation.get("action")
        inputs = operation.get("inputs")
        outputs = operation.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise ValueError(f"Operation {index} must contain input/output lists")
        if action == "keep":
            if len(inputs) != 1 or outputs != inputs:
                raise ValueError("keep must map one A1 attribute to the same A2 attribute")
            kept.extend(outputs)
        elif action == "drop":
            if len(inputs) != 1 or outputs:
                raise ValueError("drop must map one A1 attribute to no A2 attribute")
            dropped.extend(inputs)
        elif action == "add":
            if inputs or len(outputs) != 1:
                raise ValueError("add must map no A1 input to one A2 output")
        elif action == "replace":
            if len(inputs) != 1 or len(outputs) != 1 or inputs == outputs:
                raise ValueError("replace must map one A1 input to one different A2 output")
        elif action == "split":
            if len(inputs) != 1 or len(outputs) < 2:
                raise ValueError("split must map one A1 input to two or more A2 outputs")
        elif action == "merge":
            if len(inputs) < 2 or len(outputs) != 1:
                raise ValueError("merge must map two or more A1 inputs to one A2 output")
        else:
            raise ValueError(f"Unsupported refinement action: {action!r}")
        for name in outputs:
            output_counts[name] = output_counts.get(name, 0) + 1

    if set(output_counts) != set(final_names) or any(count != 1 for count in output_counts.values()):
        raise ValueError("Operations must produce every final A2 attribute exactly once")
    return {
        "mode": VECTOR_GENERATION_MODE,
        "kept_attributes": [name for name in final_names if name in set(kept)],
        "generated_attributes": final_names,
        "dropped_attributes": dropped,
        "operations": operations,
    }


def validate_a1_vectors(
    vectors: Mapping[str, Any], activities: List[str], required_attributes: List[str],
) -> Dict[str, Dict[str, str]]:
    """Validate the complete A1 matrix used only as semantic context."""
    if not isinstance(vectors, Mapping) or set(vectors) != set(activities):
        raise ValueError("A1 vectors must cover exactly the requested activities")
    cleaned: Dict[str, Dict[str, str]] = {}
    common_names = None
    for activity in activities:
        values = vectors[activity]
        if not isinstance(values, Mapping):
            raise ValueError(f"A1 vector for {activity!r} must be an object")
        names = set(map(str, values))
        common_names = names if common_names is None else common_names
        if names != common_names:
            raise ValueError("All A1 activity vectors must use the same attributes")
        cleaned[activity] = {}
        for name, raw_value in values.items():
            value = str(raw_value).strip()
            if value not in VALID_LEVELS:
                raise ValueError(f"Invalid A1 value for {activity}.{name}: {value}")
            cleaned[activity][str(name)] = value
    missing = sorted(set(required_attributes) - (common_names or set()))
    if missing:
        raise ValueError(f"A1 vectors lack attributes required by refinement operations: {missing}")
    return cleaned


def load_a1_ontology_from_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Recover the A1 ontology stored with planner evidence when available."""
    meta = plan.get("_meta")
    if not isinstance(meta, Mapping):
        return []
    evidence_value = meta.get("refinement_evidence")
    if isinstance(evidence_value, str) and evidence_value.strip():
        evidence_path = Path(evidence_value).expanduser()
        if evidence_path.is_file():
            evidence = load_json(str(evidence_path))
            ontology = evidence.get("a1_ontology") if isinstance(evidence, Mapping) else None
            if isinstance(ontology, list):
                return ontology
    ontology_value = meta.get("a1_ontology_json")
    if isinstance(ontology_value, str) and ontology_value.strip():
        ontology_path = Path(ontology_value).expanduser()
        if ontology_path.is_file():
            payload = load_json(str(ontology_path))
            ontology = payload.get("attributes") if isinstance(payload, Mapping) else payload
            if isinstance(ontology, list):
                return ontology
    return []


def assemble_full_vectors(
    *, plan: Mapping[str, Any], activities: List[str], llm_output: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, str]]:
    """Require a complete LLM matrix; never fill missing columns with A1 values."""
    operation_generation_spec(plan)
    if llm_output is None:
        raise ValueError("Full A2 generation requires a complete LLM response")
    return validate_llm_output(llm_output, activities, get_attribute_names(plan))


def assemble_incremental_vectors(
    *, plan: Mapping[str, Any], activities: List[str], a1_vectors: Mapping[str, Any],
    llm_output: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, str]]:
    """Compatibility entry point: A1 values are ignored, and full A2 is required."""
    return assemble_full_vectors(plan=plan, activities=activities, llm_output=llm_output)


def load_activities_from_json(path: str) -> List[str]:
    """
    Load activity names from a JSON file.

    Supported formats:
    1) ["lying", "sitting", ...]
    2) {"activities": ["lying", ...]}
    3) {"all_classes": ["lying", ...]}
    4) {"lying": {...}, "sitting": {...}}  # keys are activity names
    """
    data = load_json(path)

    if isinstance(data, list):
        activities = [str(x).strip() for x in data if str(x).strip()]
    elif isinstance(data, Mapping):
        if isinstance(data.get("activities"), list):
            activities = [str(x).strip() for x in data["activities"] if str(x).strip()]
        elif isinstance(data.get("all_classes"), list):
            activities = [str(x).strip() for x in data["all_classes"] if str(x).strip()]
        else:
            activities = [str(x).strip() for x in data.keys() if str(x).strip()]
    else:
        raise ValueError("--activities_json must be a list or JSON object.")

    return deduplicate_preserve_order(activities)


def deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_prompt(
    dataset_name: str,
    activities: List[str],
    plan: Mapping[str, Any],
    *,
    a1_vectors: Mapping[str, Mapping[str, str]] | None = None,
    a1_ontology: List[Mapping[str, Any]] | None = None,
    generation_spec: Mapping[str, Any] | None = None,
) -> str:
    spec = dict(generation_spec or operation_generation_spec(plan))
    final_schema = semantic_attributes(plan)
    generated_names = get_attribute_names(plan)
    plan_context = {
        "decision_reason": plan.get("decision_reason", ""),
        "semantic_redundancy_review": plan.get("semantic_redundancy_review", ""),
        "seen_classes": (plan.get("_meta") or {}).get("seen", []),
        "unseen_classes": (plan.get("_meta") or {}).get("unseen", []),
        "operations": spec["operations"],
    }

    return f"""
You are an expert in biomechanics, wearable IMU signal interpretation, and activity-level zero-shot human activity recognition.

Task:
Generation policy: {VECTOR_GENERATION_MODE}
Apply the refinement plan and generate the COMPLETE A2 matrix in ONE response:
all requested activities and ALL final A2 attributes, including keep operations.
Reassess every value using the final A2 definitions. No A1 values will be copied or
filled in by local code. A regenerated value may equal A1 when semantically justified.

These prototypes will later be used in a zero-shot HAR pipeline:
- attribute classifiers are trained to predict semantic attributes from wearable IMU signals;
- activity classes are recognized by matching predicted attributes to class-level A2 prototypes;
- the prototypes must describe activity-level motion semantics, not individual sensor samples.

Dataset:
{dataset_name}

Activities to annotate:
{json.dumps(activities, ensure_ascii=False)}

Important:
- The top-level JSON keys must exactly match the activity names in "Activities to annotate".
- Do not rename, translate, abbreviate, merge, omit, or add activities.
- Assign attributes using the same semantic principles for all activities.
- The plan's seen/unseen roles and operation reasons explain why refinement was requested.
  Use them to solve the named semantic ambiguity, but do not use or assume sensor samples.
- Generate one class-level prototype for each activity, not sample-level predictions.

Original A1 ontology:
{json.dumps(a1_ontology or [], ensure_ascii=False, indent=2)}

Original A1 class-level vectors:
{json.dumps(a1_vectors or {}, ensure_ascii=False, indent=2)}

Refinement rationale and operation lineage:
{json.dumps(plan_context, ensure_ascii=False, indent=2)}

Complete final A2 semantic schema:
{json.dumps(final_schema, ensure_ascii=False, indent=2)}

Value-generation policy:
- keep: preserve the concept's definition and level meanings, but independently reassign
  its values for every activity; existing A1 assignments are not fixed constraints;
- drop: omit the A1 attribute from A2;
- add/replace/split/merge: assign each final attribute from its A2 definition and rationale;
- A1 values are historical context only, never mandatory values or soft numeric anchors;
- do not mechanically copy, merge, take a maximum/minimum, or vote over A1 values.

Attributes whose concepts are kept (values must still be regenerated):
{json.dumps(spec['kept_attributes'], ensure_ascii=False)}

Attributes removed from A2:
{json.dumps(spec['dropped_attributes'], ensure_ascii=False)}

The complete attributes the LLM must generate:
{json.dumps(generated_names, ensure_ascii=False)}

Semantic definitions for all final attributes:
{json.dumps(final_schema, ensure_ascii=False, indent=2)}

How to use the A2 schema:
- Use each final attribute's A2 definition and Low/Medium/High descriptions as the
  authoritative value scale.
- The complete final attribute list is the exact output whitelist, including kept concepts.
  Output no dropped fields.
- Read every relevant operation reason. For split/replace/add/merge, explicitly realize the
  intended contrast between target_unseen and contrast_unseen when it is physically justified.
- Use seen_anchors as semantic calibration references, not as recording-level evidence.
- If a refined concept intentionally includes a sensor position, modality or axis in its
  definition, follow that scope consistently rather than removing it.
- Distinct physical concepts may be statistically correlated. Do not force independent level patterns or invent differences to remove correlation.

Assignment principles:
1. Values must be based on biomechanical and inertial-motion patterns that can plausibly be reflected in wearable IMU signals.
2. Relevant motion evidence may include overall motion magnitude, rhythmicity, periodicity, impact, posture change, vertical displacement, spatial spread, movement smoothness, sustained translation, rotation, and temporal regularity.
3. Do not rely on visual appearance, object identity, room context, environmental layout, or high-level task goals unless they would produce observable inertial-motion differences.
4. Do not let dataset-specific placement assumptions override the explicit A2 meanings.
5. Follow any explicit body-position, modality or axis scope in the final A2 definition; do not invent additional scope.
6. Assign values according to the abstract physical concept and generally observable human motion patterns.
7. Low, Medium, and High must be ordinal and consistent: Low < Medium < High must reflect an increasing degree of the attribute according to the A2 schema.
8. Do not mechanically map activity labels to values. Compare activities by their expected motion patterns.
9. Clearly different activities should not become identical across all A2 attributes when the schema supports meaningful biomechanical distinctions.
10. Do not fabricate differences. If two activities are genuinely similar for an attribute, assign the same level.

Internal consistency checks before output:
- Every activity must have exactly ALL attributes in "The complete attributes the LLM must generate".
- Every value must be exactly one of: "Low", "Medium", "High".
- Every attribute value must follow its A2 level definitions rather than generic intuition alone.
- The same attribute must be interpreted consistently across all activities.
- If an attribute has meaningful variation across activities, avoid assigning a constant value to all activities.
- If an attribute is genuinely constant under its definition, keep it constant rather than inventing artificial differences.
- Check whether similar activities are differentiated only when the A2 schema provides a biomechanically justified basis.
- Check whether highly different activities remain unnecessarily identical across the entire A2 vector; if so, adjust only when supported by the schema.

Output rules:
- Output JSON only.
- Do not include explanations, markdown, comments, confidence scores, reasoning traces, or extra fields.
- Do not reveal chain-of-thought. Perform the reasoning and consistency checks internally, then output only the final JSON.
- The output must be directly parseable by json.loads().

Output schema:
{{
  "<activity name>": {{
    "<final A2 attribute name>": "Low"
  }}
}}
""".strip()


def validate_llm_output(
    output: Mapping[str, Any],
    activities: List[str],
    expected_attribute_names: List[str],
) -> Dict[str, Dict[str, str]]:
    """Validate and normalize the LLM output."""
    if not isinstance(output, Mapping):
        raise ValueError("LLM output must be a JSON object.")

    expected_activity_set = set(activities)
    output_activity_set = set(str(k) for k in output.keys())

    missing_activities = [a for a in activities if a not in output]
    extra_activities = sorted(output_activity_set - expected_activity_set)
    if missing_activities or extra_activities:
        raise ValueError(
            f"Activity key mismatch. Missing={missing_activities}, extra={extra_activities}"
        )

    expected_attrs = set(expected_attribute_names)
    cleaned: Dict[str, Dict[str, str]] = {}

    for activity in activities:
        values = output[activity]
        if not isinstance(values, Mapping):
            raise ValueError(f"Activity '{activity}' must map to an attribute object.")

        got_attrs = set(str(k) for k in values.keys())
        if got_attrs != expected_attrs:
            raise ValueError(
                f"Activity '{activity}' attribute mismatch. "
                f"Missing={sorted(expected_attrs - got_attrs)}, "
                f"extra={sorted(got_attrs - expected_attrs)}"
            )

        cleaned[activity] = {}
        for attr in expected_attribute_names:
            value = str(values[attr]).strip()
            if value not in VALID_LEVELS:
                raise ValueError(
                    f"Invalid level for {activity}.{attr}: {value}. "
                    f"Expected one of {sorted(VALID_LEVELS)}."
                )
            cleaned[activity][attr] = value

    return cleaned


def build_validation_report(
    vectors: Mapping[str, Mapping[str, str]],
    activities: List[str],
    attribute_names: List[str],
) -> Dict[str, Any]:
    """Create a non-filtering audit report for transparency."""
    attr_reports: Dict[str, Any] = {}
    duplicate_vectors: Dict[str, List[str]] = {}

    for attr in attribute_names:
        values = [vectors[activity][attr] for activity in activities]
        distribution = {level: values.count(level) for level in sorted(VALID_LEVELS)}
        attr_reports[attr] = {
            "distribution_all_activities": distribution,
            "num_levels_used": len(set(values)),
            "is_constant_all_activities": len(set(values)) <= 1,
        }

    signature_to_activities: Dict[str, List[str]] = {}
    for activity in activities:
        signature = "|".join(vectors[activity][attr] for attr in attribute_names)
        signature_to_activities.setdefault(signature, []).append(activity)

    for signature, acts in signature_to_activities.items():
        if len(acts) > 1:
            duplicate_vectors[signature] = acts

    return {
        "policy": {
            "filtering": False,
            "description": (
                "This report only checks distribution and duplicate full vectors. "
                "No attribute is removed in this script."
            ),
        },
        "num_activities": len(activities),
        "num_attributes": len(attribute_names),
        "attribute_reports": attr_reports,
        "duplicate_full_vectors": duplicate_vectors,
    }


def response_provenance_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.name + ".generation.json")


def record_generation_response(raw_path: Path, plan_path: Path, a1_path: Path | None) -> None:
    """Mark a response as a complete-matrix request before validation/recovery."""
    dump_json({
        "mode": VECTOR_GENERATION_MODE,
        "a2_plan_sha256": sha256(plan_path),
        "raw_response_sha256": sha256(raw_path),
        "a1_json": str(a1_path.resolve()) if a1_path else None,
        "a1_sha256": sha256(a1_path) if a1_path else None,
    }, str(response_provenance_path(raw_path)))


def validate_generation_response(
    raw_path: Path, plan_path: Path, a1_path: Path | None = None,
) -> dict:
    source = load_json(str(response_provenance_path(raw_path)))
    if source.get("mode") != VECTOR_GENERATION_MODE:
        raise ValueError("Response was not generated with the complete A2 policy")
    if source.get("a2_plan_sha256") != sha256(plan_path):
        raise ValueError("Response plan hash mismatch")
    if source.get("raw_response_sha256") != sha256(raw_path):
        raise ValueError("Raw response hash mismatch")
    source_a1 = a1_path or (Path(source["a1_json"]) if source.get("a1_json") else None)
    if source_a1 and source.get("a1_sha256") != sha256(source_a1):
        raise ValueError("Response A1 context hash mismatch")
    return source


def build_generation_report(
    vectors, activities, *, plan, plan_path: Path, vector_path: Path,
    raw_path: Path, a1_path: Path | None = None,
) -> dict:
    source = validate_generation_response(raw_path, plan_path, a1_path)
    spec = operation_generation_spec(plan)
    report = build_validation_report(vectors, activities, get_attribute_names(plan))
    report["vector_source"] = {
        **source,
        "a2_plan_json": str(plan_path.resolve()),
        "vectors_sha256": sha256(vector_path),
        "copied_keep_attributes": [],
        "conceptually_kept_attributes": spec["kept_attributes"],
        "llm_generated_attributes": spec["generated_attributes"],
        "dropped_a1_attributes": spec["dropped_attributes"],
        "llm_called": True,
    }
    return report


def validate_saved_generation(
    *, vector_path: Path, report_path: Path, plan_path: Path, raw_path: Path,
    a1_path: Path | None = None,
) -> None:
    """Reject incremental/fallback artifacts and stale full-generation checkpoints."""
    report = load_json(str(report_path))
    source = report.get("vector_source", {})
    response = validate_generation_response(raw_path, plan_path, a1_path)
    if any(source.get(key) != value for key, value in response.items()):
        raise ValueError("Vector report does not match the full-generation response")
    if source.get("vectors_sha256") != sha256(vector_path):
        raise ValueError("Saved vector hash mismatch")
    if source.get("copied_keep_attributes") != [] or source.get("llm_called") is not True:
        raise ValueError("Every A2 value must be generated by the LLM")


def resolve_a1_source(
    plan: Mapping[str, Any], *, explicit_path: str = "", activities_json: str = "",
) -> Path | None:
    """Resolve optional historical A1 context, never an output value source."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    meta = plan.get("_meta")
    if isinstance(meta, Mapping) and isinstance(meta.get("a1_json"), str):
        candidates.append(meta["a1_json"])
    # Existing runners already pass the A1 vector JSON as --activities_json.
    if activities_json:
        candidates.append(activities_json)
    for value in candidates:
        path = Path(value).expanduser()
        if path.is_file():
            try:
                payload = load_json(str(path))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping) and payload and all(
                isinstance(item, Mapping) for item in payload.values()
            ):
                return path.resolve()
    return None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the complete A2 matrix in one request, reassigning every value."
    )
    parser.add_argument("--dataset_name", default="pamap2")
    parser.add_argument("--a2_plan_json", required=True)
    parser.add_argument(
        "--a1_json", "--a1-json", default="",
        help="Initial A1 activity vectors; defaults to plan provenance or --activities_json",
    )
    parser.add_argument("--output_json", required=True)

    activity_group = parser.add_mutually_exclusive_group(required=True)
    activity_group.add_argument(
        "--activities",
        help="Comma- or pipe-separated activity names, e.g. 'lying,sitting,standing'.",
    )
    activity_group.add_argument(
        "--activities_json",
        help=(
            "JSON file containing activity names. Supported formats: list, "
            "{'activities': [...]}, {'all_classes': [...]}, or dict whose keys are activity names."
        ),
    )

    parser.add_argument("--prompt_out", default="")
    parser.add_argument("--raw_response_out", default="")
    parser.add_argument("--report_json", default="")
    parser.add_argument(
        "--api_format", "--api-format", choices=("openai", "gemini"),
        default=os.getenv("LLM_API_FORMAT", "openai"),
    )
    parser.add_argument(
        "--base_url", "--base-url",
        default=os.getenv("LLM_BASE_URL", "https://www.kamiapi.top/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", os.getenv("MODEL_NAME", "gpt-5.6-sol")),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--stream", action="store_true",
        help="Use streaming for OpenAI-compatible APIs to avoid proxy read timeouts.",
    )
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args(argv)
    if args.stream and args.api_format != "openai":
        parser.error("--stream is supported only with --api-format openai")

    plan = load_json(args.a2_plan_json)
    attribute_names = get_attribute_names(plan)
    generation_spec = operation_generation_spec(plan)

    if args.activities_json:
        activities = load_activities_from_json(args.activities_json)
    else:
        activities = deduplicate_preserve_order(parse_activity_list(args.activities))

    if not activities:
        raise ValueError("No activities were provided.")

    a1_source = resolve_a1_source(
        plan, explicit_path=args.a1_json, activities_json=args.activities_json or ""
    )
    a1_vectors: Dict[str, Dict[str, str]] | None = None
    if a1_source is not None:
        meta = plan.get("_meta")
        expected_hash = meta.get("a1_sha256") if isinstance(meta, Mapping) else None
        actual_hash = sha256(a1_source)
        if isinstance(expected_hash, str) and expected_hash and expected_hash != actual_hash:
            raise ValueError(
                f"A1 vector hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        required_a1 = sorted({
            source
            for operation in generation_spec["operations"]
            for source in operation.get("inputs", [])
        })
        a1_vectors = validate_a1_vectors(
            load_json(str(a1_source)), activities, required_a1
        )

    output_path = Path(args.output_json)
    prompt_out = args.prompt_out or str(output_path.with_suffix(".prompt.txt"))
    raw_response_out = args.raw_response_out or str(output_path.with_suffix(".raw_response.txt"))
    report_json = args.report_json or str(output_path.with_suffix(".report.json"))
    outputs = [Path(value).resolve() for value in (
        output_path, prompt_out, raw_response_out, report_json,
        response_provenance_path(Path(raw_response_out)),
    )]
    inputs = {Path(args.a2_plan_json).resolve()}
    if a1_source:
        inputs.add(a1_source.resolve())
    if args.activities_json:
        inputs.add(Path(args.activities_json).resolve())
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        raise ValueError("Generation outputs must be distinct and must not overwrite inputs")

    prompt = build_prompt(
        dataset_name=args.dataset_name,
        activities=activities,
        plan=plan,
        a1_vectors=a1_vectors,
        a1_ontology=load_a1_ontology_from_plan(plan),
        generation_spec=generation_spec,
    )

    write_text(prompt, prompt_out)

    if args.dry_run:
        print(prompt)
        print(f"\nSaved vector-completion prompt to: {prompt_out}")
        return

    raw = call_llm(
        messages=[{"role": "user", "content": prompt}],
        api_format=args.api_format,
        api_key=resolve_api_key(args.api_format, args.base_url),
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        ssl_no_verify=False,
        stream=args.stream,
    )
    write_text(raw, raw_response_out)
    record_generation_response(Path(raw_response_out), Path(args.a2_plan_json), a1_source)

    llm_output = extract_json_object(raw)
    cleaned = assemble_full_vectors(
        plan=plan, activities=activities, llm_output=llm_output,
    )

    dump_json(cleaned, args.output_json)
    report = build_generation_report(
        cleaned, activities, plan=plan, plan_path=Path(args.a2_plan_json),
        vector_path=output_path, raw_path=Path(raw_response_out), a1_path=a1_source,
    )
    dump_json(report, report_json)

    print(f"Saved complete A2 activity vectors to: {args.output_json}")
    print(f"Saved prompt to: {prompt_out}")
    print(f"Saved raw LLM response to: {raw_response_out}")
    print(f"Saved non-filtering validation report to: {report_json}")
    print(f"Generated {len(activities)} activities × {len(attribute_names)} A2 attributes.")


if __name__ == "__main__":
    main()
