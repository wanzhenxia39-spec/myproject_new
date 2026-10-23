#!/usr/bin/env python3
"""Split-specific, semantic-only A1 -> A2 attribute refinement for zero-shot HAR."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np
import pandas as pd

from zshar.datasets import load_dataset_spec
from zshar.features.groups import expand_feature_groups, load_groups, resolve_feature_groups
from zshar.llm.transport import call_llm, resolve_api_key as resolve_transport_api_key


DEFAULT_MIN_ATTRIBUTES = 9
DEFAULT_MAX_ATTRIBUTES = 15
REFINEMENT_STRATEGY = "full_a1_semantic_refinement_v3"
LEVELS = {"Low", "Medium", "High"}
SEMANTIC_FIELDS = {"name", "definition", "levels"}

# Coarse capabilities are shown to the LLM. Exact feature columns and selectors are not.
GROUP_PHYSICAL_VOCAB = {
    "G1": "static gravity-relative orientation and mean posture",
    "G2": "sustained or directional rotational motion",
    "G3": "overall motion magnitude and energy",
    "G4": "temporal variability of motion intensity",
    "G5": "gradual temporal trend or sustained change",
    "G6": "repetition rate, dominant frequency and periodicity",
    "G7": "distribution of motion energy across frequency bands",
    "G8": "spectral complexity or frequency dispersion",
    "G9": "sustained rapid signal change related to smoothness",
    "G10": "peak rapid change related to impulsiveness or impact",
    "G11": "temporal regularity, recurrence and predictability",
    "G12": "variation of gravity-relative posture over time",
    "G13": "short-window translational-motion or velocity proxy",
    "G14": "multiaxial motion structure and acceleration-rotation relation",
}


def parse_activity_list(text: str) -> List[str]:
    if not text:
        return []
    separator = "|" if "|" in text else ","
    return [item.strip() for item in text.split(separator) if item.strip()]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def extract_json_object(text: str) -> Any:
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
    return resolve_transport_api_key(api_format, base_url)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attribute_names(a1_vectors: Mapping[str, Mapping[str, str]]) -> List[str]:
    if not isinstance(a1_vectors, dict) or not a1_vectors:
        raise ValueError("A1 attribute JSON is empty")
    first = next(iter(a1_vectors.values()))
    if not isinstance(first, dict) or not first:
        raise ValueError("A1 class vectors must be nonempty attribute dictionaries")
    return list(first)


def _json_field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if isinstance(value, str):
            return json.loads(value)
        return value
    raise ValueError(f"GAP CSV is missing structured field {names}")


def _finite_float(row: Mapping[str, Any], name: str, *, allow_nan: bool = False) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GAP CSV has no valid {name}") from exc
    if math.isnan(value) and allow_nan:
        return None
    if not math.isfinite(value):
        raise ValueError(f"GAP CSV has no finite {name}")
    return value


def resolve_split_from_gap_csv(gap_csv, seen, unseen, split_index):
    """Load split membership and precomputed D_kNN/U_NN diagnostics from the CSV."""
    frame = pd.read_csv(gap_csv)
    if "split_index" not in frame or frame["split_index"].duplicated().any():
        raise ValueError("GAP split_index must exist and be unique")
    if split_index is not None:
        matched = frame[frame["split_index"] == int(split_index)]
    elif unseen:
        matched = frame[frame["unseen_classes"].map(
            lambda value: set(parse_activity_list(str(value))) == set(unseen)
        )]
    else:
        raise ValueError("Specify --split-index, --unseen, or --all-splits")
    if len(matched) != 1:
        raise ValueError("Could not locate exactly one requested split in --dknn-csv")

    row = matched.iloc[0].to_dict()
    row_seen = parse_activity_list(str(row["seen_classes"]))
    row_unseen = parse_activity_list(str(row["unseen_classes"]))
    if ((seen and set(seen) != set(row_seen)) or
            (unseen and set(unseen) != set(row_unseen))):
        raise ValueError("Explicit seen/unseen classes disagree with GAP split")
    if (not row_seen or not row_unseen or set(row_seen) & set(row_unseen) or
            len(set(row_seen)) != len(row_seen) or len(set(row_unseen)) != len(row_unseen)):
        raise ValueError("GAP split must contain nonempty, unique, disjoint class lists")

    d_per = _json_field(row, "D_kNN_per_unseen_json", "D_kNN_per_unseen")
    u_per = _json_field(row, "U_NN_per_unseen_json", "U_NN_per_unseen")
    nearest = _json_field(row, "nearest_seen_topk_json")
    if not isinstance(d_per, dict) or set(d_per) != set(row_unseen):
        raise ValueError("D_kNN_per_unseen must cover exactly the unseen classes")
    if len(row_unseen) > 1 and (not isinstance(u_per, dict) or set(u_per) != set(row_unseen)):
        raise ValueError("U_NN_per_unseen must cover exactly the unseen classes")
    if not isinstance(nearest, dict) or set(nearest) != set(row_unseen):
        raise ValueError("nearest_seen_topk must cover exactly the unseen classes")

    top_k = int(row.get("top_k", 2))
    effective_k = int(row.get("effective_k", min(top_k, len(row_seen))))
    metric = str(row.get("distance_metric", "mad"))
    if top_k < 1 or effective_k < 1 or metric not in {"normalized_ordinal", "mad", "euclidean"}:
        raise ValueError("Invalid top_k/effective_k/distance_metric in GAP CSV")

    semantic_gap = {
        "D_kNN": _finite_float(row, "D_kNN"),
        "D_kNN_per_unseen": {str(k): float(v) for k, v in d_per.items()},
        "U_NN": _finite_float(row, "U_NN", allow_nan=True),
        "U_NN_per_unseen": {str(k): float(v) for k, v in u_per.items()},
        "nearest_seen_topk": nearest,
        "top_k": top_k,
        "effective_k": effective_k,
        "distance_metric": metric,
    }
    return {
        "split_index": int(row["split_index"]),
        "seen": row_seen,
        "unseen": row_unseen,
        "semantic_gap": semantic_gap,
        "a1_sha256": row.get("a1_sha256") if isinstance(row.get("a1_sha256"), str) else None,
    }


def validate_a1_ontology(ontology, names):
    entries = ontology.get("attributes") if isinstance(ontology, dict) else ontology
    if entries is None and isinstance(ontology, dict):
        entries = ontology
    if isinstance(entries, dict):
        entries = [{"name": name, **value} for name, value in entries.items()
                   if isinstance(value, dict)]
    if not isinstance(entries, list) or len(entries) != len(names):
        raise ValueError("A1 ontology must contain one definition per A1 attribute")
    by_name = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("Invalid A1 ontology entry")
        definition = entry.get("definition", entry.get("meaning"))
        levels = entry.get("levels")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError("Each A1 attribute needs a physical definition")
        if (not isinstance(levels, dict) or set(levels) != LEVELS or
                any(not isinstance(value, str) or not value.strip() for value in levels.values())):
            raise ValueError("Each A1 attribute needs Low/Medium/High definitions")
        if entry["name"] in by_name:
            raise ValueError("Duplicate A1 ontology name")
        by_name[entry["name"]] = {
            "name": entry["name"], "definition": definition, "levels": levels,
        }
    if set(by_name) != set(names):
        raise ValueError("A1 ontology names do not match A1 vectors")
    return [by_name[name] for name in names]


def build_split_evidence(dataset, gap_info, vectors, ontology):
    """Combine A1 semantics with diagnostics already computed by gap_analysis.py."""
    names = attribute_names(vectors)
    return {
        "schema_version": REFINEMENT_STRATEGY,
        "task": {
            "dataset": dataset,
            "outer_split_index": gap_info["split_index"],
            "seen_classes": gap_info["seen"],
            "unseen_classes": gap_info["unseen"],
        },
        "a1_ontology": validate_a1_ontology(ontology, names),
        "a1_class_prototypes": vectors,
        "semantic_gap": gap_info["semantic_gap"],
        "data_provenance": {
            "uses_real_unseen_identities": True,
            "uses_real_unseen_semantic_prototypes": True,
            "uses_target_recordings": False,
            "uses_test_predictions": False,
            "diagnostics_source": "reused_from_gap_csv",
        },
    }


def compact_physical_capabilities(dataset, feature_names, config):
    """List only groups backed by real columns; do not infer selector profiles."""
    partition = resolve_feature_groups(feature_names, config, dataset=dataset)
    groups = {}
    for group_id, group in config["groups"].items():
        if partition["group_to_features"].get(group_id):
            groups[group_id] = {
                "name": group.get("name", group_id),
                "physical_capability": GROUP_PHYSICAL_VOCAB[group_id],
            }
    return {
        "feature_groups": groups,
        "sensor_inventory": config["dataset_sensors"][dataset],
    }


def full_prompt_evidence(evidence, capabilities):
    return {
        "task": evidence["task"],
        "a1_ontology": evidence["a1_ontology"],
        "a1_class_prototypes": evidence["a1_class_prototypes"],
        "semantic_gap": evidence["semantic_gap"],
        "imu_physical_capabilities": capabilities["feature_groups"],
        "sensor_inventory": capabilities["sensor_inventory"],
    }


def build_system_prompt():
    return """You are an expert in wearable-IMU human activity recognition, biomechanics,
zero-shot learning, and interpretable semantic attribute design.

Create one compact, split-specific A2 semantic attribute set for a zero-shot activity
recognition experiment. Use only the supplied class-level semantics and IMU capability
summary. Do not assume recordings, sample statistics, predictions, classifier accuracy, or
unlisted features. Think through the comparison internally and return JSON only.""".strip()


def build_prompt(evidence, capabilities, *, min_attributes=DEFAULT_MIN_ATTRIBUTES,
                 max_attributes=DEFAULT_MAX_ATTRIBUTES):
    payload = full_prompt_evidence(evidence, capabilities)
    task = payload["task"]
    return f"""Refine the initial A1 semantic attributes for this zero-shot split.

1. Zero-shot task and class split
{json.dumps(task, ensure_ascii=False, indent=2)}

2. Complete A1 ontology
Learn the exact meaning and Low/Medium/High ordering of every initial attribute from this
ontology. Preserve a kept attribute's name, definition and levels exactly.
{json.dumps(payload['a1_ontology'], ensure_ascii=False, indent=2)}

3. Complete A1 class prototype matrix
Compare all activity pairs directly. Focus especially on semantically similar pairs whose
A1 vectors differ in only one or two coordinates. When physically justified, add, split or
replace a dimension so that such a pair gains another clear and transferable distinction.
Do not manufacture differences when the supplied IMU capabilities cannot support them.
{json.dumps(payload['a1_class_prototypes'], ensure_ascii=False, indent=2)}

4. Precomputed A1 coverage and unseen-pair diagnostics
These values were calculated before refinement and are reused directly from the GAP CSV.
- A high D_kNN means an unseen class has weak coverage by nearby seen prototypes; prioritize
  checking whether its semantics need a better seen-supported distinction.
- A low U_NN means an unseen class is close to another unseen class; prioritize adding a
  physically meaningful distinction for that confusing unseen pair.
- nearest_seen_topk identifies possible transferable seen anchors.
Use these diagnostics as guidance, not as recognition accuracy.
{json.dumps(payload['semantic_gap'], ensure_ascii=False, indent=2)}

5. Available IMU support
Every final attribute must be plausibly supported by at least one listed G1-G14 physical
capability. Cite supporting group IDs as semantic evidence in operation reasons only;
this is a feasibility assessment, not a feature binding. Do not return group bindings,
selectors, feature names or feature indices.
{json.dumps(payload['imu_physical_capabilities'], ensure_ascii=False, indent=2)}

The dataset sensor inventory is shown only to establish what can physically be observed:
{json.dumps(payload['sensor_inventory'], ensure_ascii=False, indent=2)}

6. Refinement policy
- Compare unseen activities with one another and use seen activities as semantic anchors.
- Keep useful A1 concepts. You may keep, split, merge, drop, replace or add concepts.
- Account for every A1 input and every A2 output exactly once in operations.
- Prefer a change when it adds a supported independent contrast for a close A1 activity pair,
  improves weak seen coverage indicated by D_kNN, or separates a low-U_NN unseen pair.
- Sensor modality, body position or motion axis may be used in a refinement when that scope
  is available in the sensor inventory, has a clear physical meaning, and materially
  distinguishes a confusing activity pair. Avoid arbitrary sensor-specific renaming.
- Attribute names may be one word. Multiword names must use lowercase snake_case.
- Return between {min_attributes} and {max_attributes} attributes. Do not add attributes only
  to reach a number.
- Give a concise physical and zero-shot transfer reason for every operation. Lists named
  target_unseen and contrast_unseen may contain only unseen classes; seen_anchors may contain
  only seen classes. Copy every activity name verbatim from the supplied class split; do not
  add, remove or alter punctuation. Use empty lists when a role does not apply.

When proposing a new or refined attribute, consider not only its semantic discriminability
but also its expected predictability from the available IMU signals and extracted features.
Prefer attributes whose ordinal states correspond to distinct, observable, and stable motion
patterns that can generalize across activity classes.
For each proposed attribute, identify the supporting feature groups, explain the expected
sensor evidence, and describe potential sources of prediction ambiguity. Avoid attributes
that rely on fine-grained motion details unlikely to be reliably captured by the available
sensors.
Do not assume that semantic distinctiveness guarantees sensor-level predictability. If the
expected predictability or added discriminative value is uncertain, prefer preserving the
existing A1 attributes. Do not generate class-level attribute vectors at this stage.
Include each proposed attribute's concise assessment in its operation's reason, naming each
output attribute separately when an operation has multiple outputs. Use only listed groups
and the supplied sensor inventory; these are expected evidence, not measured performance.

7. Output contract
Return JSON only:
{{"schema_version":"2.0", "dataset":{json.dumps(task['dataset'])},
 "split_index":{json.dumps(task['outer_split_index'])}, "refinement_needed":true,
 "decision_reason":"split-specific diagnosis of A1",
 "attributes":[{{
   "name":"physical_concept",
   "definition":"physical meaning",
   "levels":{{"Low":"low degree", "Medium":"intermediate degree", "High":"high degree"}}
 }}],
 "operations":[{{"action":"keep|split|merge|drop|replace|add",
   "inputs":["source_name"], "outputs":["output_name"],
   "reason":"physical distinction and transfer basis; per output: supporting groups, expected sensor evidence, prediction ambiguity",
   "target_unseen":[], "contrast_unseen":[], "seen_anchors":[]}}],
 "semantic_redundancy_review":"review of conceptual overlap"}}

Before returning, check that the final attributes are physically distinct, ordinal,
supported in principle by the listed capabilities, and completely accounted for by the
operations. Do not return feature-group binding fields.""".strip()


def build_messages(evidence, capabilities, *, min_attributes=DEFAULT_MIN_ATTRIBUTES,
                   max_attributes=DEFAULT_MAX_ATTRIBUTES):
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_prompt(
            evidence, capabilities,
            min_attributes=min_attributes, max_attributes=max_attributes,
        )},
    ]


def _validate_attribute_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Every A2 attribute needs a nonempty name")
    if re.fullmatch(r"[a-z][a-z0-9]*", name):
        return
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", name):
        return
    raise ValueError("Multiword A2 attribute names must use lowercase snake_case")


def _activity_token(value: str) -> str:
    """Normalize harmless presentation differences for exact vocabulary recovery."""
    return "".join(character for character in value.casefold() if character.isalnum())


def canonicalize_operation_activity_roles(plan, *, seen, unseen):
    """Repair only uniquely identifiable case/space/punctuation activity-name drift.

    Semantic substitutions and fuzzy spelling corrections are intentionally excluded.
    Values that cannot be mapped uniquely remain unchanged and are rejected by the
    normal semantic-plan validator.
    """
    repairs = []
    operations = plan.get("operations") if isinstance(plan, dict) else None
    if not isinstance(operations, list):
        return repairs
    for operation_index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            continue
        for field, allowed in (
            ("target_unseen", unseen),
            ("contrast_unseen", unseen),
            ("seen_anchors", seen),
        ):
            values = operation.get(field)
            if not isinstance(values, list):
                continue
            by_token = {}
            for activity in allowed:
                by_token.setdefault(_activity_token(activity), []).append(activity)
            normalized = []
            for value in values:
                replacement = value
                if isinstance(value, str) and value not in allowed:
                    candidates = by_token.get(_activity_token(value), [])
                    if len(candidates) == 1:
                        replacement = candidates[0]
                        repairs.append({
                            "operation_index": operation_index,
                            "field": field,
                            "from": value,
                            "to": replacement,
                        })
                normalized.append(replacement)
            operation[field] = normalized
    return repairs


def validate_semantic_plan(plan, *, a1_ontology, a1_names, seen, unseen, dataset,
                           split_index, min_attributes=DEFAULT_MIN_ATTRIBUTES,
                           max_attributes=DEFAULT_MAX_ATTRIBUTES):
    """Validate semantic schema and lineage only; no feature binding is performed."""
    if not isinstance(plan, dict) or plan.get("schema_version") != "2.0":
        raise ValueError("Expected refinement schema_version 2.0")
    if plan.get("dataset") != dataset or plan.get("split_index") != split_index:
        raise ValueError("Refinement dataset/split does not match input")
    attributes = plan.get("attributes")
    if not isinstance(attributes, list) or not min_attributes <= len(attributes) <= max_attributes:
        raise ValueError("A2 attribute count outside allowed bounds")

    names, definitions = [], []
    for attribute in attributes:
        if not isinstance(attribute, dict) or set(attribute) != SEMANTIC_FIELDS:
            raise ValueError("Each A2 attribute must contain only name, definition and levels")
        _validate_attribute_name(attribute["name"])
        definition = attribute["definition"]
        levels = attribute["levels"]
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError("Each A2 attribute needs a definition")
        if (not isinstance(levels, dict) or set(levels) != LEVELS or
                any(not isinstance(value, str) or not value.strip() for value in levels.values())):
            raise ValueError("Each A2 attribute must define Low/Medium/High")
        names.append(attribute["name"])
        definitions.append(" ".join(definition.lower().split()))
    if len(set(names)) != len(names) or len(set(definitions)) != len(definitions):
        raise ValueError("Duplicate A2 names or identical definitions")

    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("Record lineage decisions for every A1 input and A2 output")
    inputs, outputs = [], []
    by_name = {attribute["name"]: attribute for attribute in attributes}
    source_ontology = {attribute["name"]: attribute for attribute in a1_ontology}
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each operation must be an object")
        action = operation.get("action")
        sources, targets = operation.get("inputs"), operation.get("outputs")
        if not isinstance(sources, list) or not isinstance(targets, list):
            raise ValueError("Operation inputs/outputs must be lists")
        if any(source not in a1_names for source in sources) or any(target not in names for target in targets):
            raise ValueError("Unknown operation input/output")
        valid_shape = (
            (action == "keep" and len(sources) == 1 and targets == sources) or
            (action == "split" and len(sources) == 1 and len(targets) >= 2) or
            (action == "merge" and len(sources) >= 2 and len(targets) == 1) or
            (action == "drop" and len(sources) == 1 and not targets) or
            (action == "replace" and len(sources) == len(targets) == 1 and sources != targets) or
            (action == "add" and not sources and len(targets) == 1)
        )
        if not valid_shape:
            raise ValueError("Invalid concept-lineage action or cardinality")
        if not isinstance(operation.get("reason"), str) or not operation["reason"].strip():
            raise ValueError("Each operation needs a reason")
        for field, allowed in (("target_unseen", unseen), ("contrast_unseen", unseen),
                               ("seen_anchors", seen)):
            values = operation.get(field, [])
            if not isinstance(values, list) or any(value not in allowed for value in values):
                raise ValueError(f"Invalid {field} in operation")
        if action == "keep" and by_name[sources[0]] != source_ontology[sources[0]]:
            raise ValueError("Kept concepts must preserve the A1 definition and levels")
        inputs.extend(sources)
        outputs.extend(targets)
    if Counter(inputs) != Counter(a1_names) or Counter(outputs) != Counter(names):
        raise ValueError("Operations must account for every A1 input and A2 output exactly once")

    changed = any(operation["action"] != "keep" for operation in operations)
    if type(plan.get("refinement_needed")) is not bool or plan["refinement_needed"] != changed:
        raise ValueError("refinement_needed must agree with conceptual operations")
    for field in ("decision_reason", "semantic_redundancy_review"):
        if not isinstance(plan.get(field), str) or not plan[field].strip():
            raise ValueError(f"Missing {field}")
    return plan


def validate_single_stage_response(plan, evidence, context):
    return validate_semantic_plan(plan, a1_ontology=evidence["a1_ontology"], **context)


def readable_messages(messages):
    return "\n\n".join(
        f"===== {message['role'].upper()} =====\n{message['content']}" for message in messages
    ) + "\n"


def check_output_strategy(output_path):
    """Prevent accidental mixing with plans created by an older planner strategy."""
    output = Path(output_path)
    stem = output.with_suffix("")
    evidence_path = Path(str(stem) + ".refinement_evidence.json")
    artifacts = list(output.parent.glob(stem.name + ".*"))
    if artifacts and (not evidence_path.is_file() or
                      load_json(evidence_path).get("schema_version") != REFINEMENT_STRATEGY):
        raise ValueError(
            f"Existing artifacts in {output.parent} use another refinement strategy. "
            "Choose a new output directory; old results are preserved."
        )


def run_split(*, dataset, a1_path, ontology_path, gap_path, gap_info, output_path,
              feature_names, feature_source, config, model="gpt-5.6-sol", temperature=0,
              min_attributes=DEFAULT_MIN_ATTRIBUTES, max_attributes=DEFAULT_MAX_ATTRIBUTES,
              dry_run=False, concepts_path=None, prompt_out=None, prompt_text_out=None,
              raw_response_out=None, api_format="openai", api_key="", base_url="",
              timeout=300, stream=False):
    """Make one semantic refinement call and validate the returned A2 schema locally."""
    check_output_strategy(output_path)
    vectors = load_json(a1_path)
    source_names = attribute_names(vectors)
    seen, unseen = gap_info["seen"], gap_info["unseen"]
    if set(vectors) != set(seen + unseen):
        raise ValueError("A1 class vocabulary does not match this GAP split")
    if any(not isinstance(values, dict) or set(values) != set(source_names) or
           any(level not in LEVELS for level in values.values()) for values in vectors.values()):
        raise ValueError("A1 vectors must share attributes with Low/Medium/High values")
    gap_hash = gap_info.get("a1_sha256")
    if gap_hash and gap_hash != _sha(a1_path):
        raise ValueError("GAP was calculated from different A1 content; regenerate the GAP CSV")

    output = Path(output_path)
    stem = output.with_suffix("")
    prompt_json = Path(prompt_out) if prompt_out else Path(str(stem) + ".prompt.json")
    prompt_text = Path(prompt_text_out) if prompt_text_out else Path(str(stem) + ".prompt.txt")
    paths = {
        "prompt_json": prompt_json,
        "prompt_text": prompt_text,
        "raw": Path(raw_response_out or str(stem) + ".raw_response.txt"),
        "evidence": Path(str(stem) + ".refinement_evidence.json"),
    }
    destinations = [output, *paths.values()]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("Output artifacts must be distinct")
    immutable_inputs = {Path(a1_path).resolve(), Path(ontology_path).resolve(), Path(gap_path).resolve()}
    if immutable_inputs & {path.resolve() for path in destinations}:
        raise ValueError("Output artifacts cannot overwrite immutable inputs")

    evidence = build_split_evidence(dataset, gap_info, vectors, load_json(ontology_path))
    capabilities = compact_physical_capabilities(dataset, feature_names, config)
    messages = build_messages(
        evidence, capabilities,
        min_attributes=min_attributes, max_attributes=max_attributes,
    )
    dump_json(evidence, paths["evidence"])
    dump_json(messages, paths["prompt_json"])
    write_text(readable_messages(messages), paths["prompt_text"])
    if dry_run and not concepts_path:
        return {
            "prompt_json": str(paths["prompt_json"]),
            "prompt_text": str(paths["prompt_text"]),
            "status": "prompt_only",
        }

    if concepts_path:
        raw_text = Path(concepts_path).read_text(encoding="utf-8")
    else:
        raw_text = call_llm(
            messages=messages,
            api_format=api_format,
            api_key=api_key or resolve_api_key(api_format, base_url),
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            ssl_no_verify=False,
            stream=stream,
        )
    if not concepts_path or Path(concepts_path).resolve() != paths["raw"].resolve():
        write_text(raw_text, paths["raw"])
    plan = extract_json_object(raw_text)
    activity_role_repairs = canonicalize_operation_activity_roles(
        plan, seen=seen, unseen=unseen
    )
    context = {
        "a1_names": source_names,
        "seen": seen,
        "unseen": unseen,
        "dataset": dataset,
        "split_index": gap_info["split_index"],
        "min_attributes": min_attributes,
        "max_attributes": max_attributes,
    }
    validate_single_stage_response(plan, evidence, context)

    meta = {
        "dataset": dataset,
        "split_index": gap_info["split_index"],
        "seen": seen,
        "unseen": unseen,
        "refinement_strategy": REFINEMENT_STRATEGY,
        "planning_stages": 1,
        "min_attributes": min_attributes,
        "max_attributes": max_attributes,
        "a1_json": str(a1_path),
        "a1_sha256": _sha(a1_path),
        "a1_ontology_json": str(ontology_path),
        "a1_ontology_sha256": _sha(ontology_path),
        "split_csv_sha256": _sha(gap_path),
        "gap_diagnostics_source": "reused_from_gap_csv",
        "refinement_evidence": str(paths["evidence"]),
        "refinement_evidence_sha256": _sha(paths["evidence"]),
        "prompt_json": str(paths["prompt_json"]),
        "prompt_json_sha256": _sha(paths["prompt_json"]),
        "prompt_text": str(paths["prompt_text"]),
        "prompt_text_sha256": _sha(paths["prompt_text"]),
        "gap_a1_hash_verified": bool(gap_hash),
        "feature_source": feature_source,
        "feature_names_sha256": hashlib.sha256(json.dumps(list(feature_names)).encode()).hexdigest(),
        "groups_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "model": model,
        "api_format": api_format,
        "base_url": base_url,
        "stream": stream,
        "response_source": str(concepts_path) if concepts_path else "live",
        "validation_decision": "validated_semantic_a2",
        "validation": "Semantic schema and complete A1-to-A2 lineage validated; no feature binding performed.",
        "activity_role_repairs": activity_role_repairs,
    }
    plan["_meta"] = meta
    dump_json(plan, output)
    return {"plan": str(output), "status": "validated_semantic_a2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    def option(name, *aliases, **kwargs):
        parser.add_argument("--" + name, "--" + name.replace("_", "-"), *aliases, **kwargs)

    option("dataset_name", default="pamap2")
    option("a1_json", required=True)
    option("dknn_csv", "--split-csv", required=True,
           help="Precomputed GAP CSV containing D_kNN, U_NN and per-unseen diagnostics")
    option("a1_ontology_json", required=True)
    option("output_json")
    option("output_root")
    option("all_splits", action="store_true")
    option("split_index", type=int)
    parser.add_argument("--seen", default="")
    parser.add_argument("--unseen", default="")
    option("feature_names_npy")
    option("groups_json")
    option("templates_only", action="store_true",
           help="Build a prompt from configured templates instead of actual feature names")
    option("concepts_json", help="Replay and validate a semantic A2 JSON response")
    option("responses_root", help="Batch replay responses under each split_N directory")
    option("prompt_out")
    option("prompt_text_out")
    option("raw_response_out")
    option("min_attributes", type=int, default=DEFAULT_MIN_ATTRIBUTES)
    option("max_attributes", type=int, default=DEFAULT_MAX_ATTRIBUTES)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", os.getenv("MODEL_NAME", "gpt-5.6-sol")))
    parser.add_argument("--temperature", type=float, default=0)
    option("api_format", choices=("gemini", "openai"), default=os.getenv("LLM_API_FORMAT", "openai"))
    option("base_url", default=os.getenv("LLM_BASE_URL", "https://www.kamiapi.top/v1"))
    option("timeout", type=float, default=300)
    option("stream", action="store_true")
    option("dry_run", action="store_true")
    args = parser.parse_args()

    if args.stream and args.api_format != "openai":
        parser.error("--stream is currently supported only with --api-format openai")
    if not DEFAULT_MIN_ATTRIBUTES <= args.min_attributes <= args.max_attributes <= DEFAULT_MAX_ATTRIBUTES:
        parser.error("Require 9 <= min-attributes <= max-attributes <= 15")
    if args.templates_only and (not args.dry_run or args.concepts_json or args.responses_root):
        parser.error("--templates-only is limited to prompt previews")
    if args.all_splits:
        if not args.output_root or any((args.output_json, args.split_index is not None,
               args.seen, args.unseen, args.concepts_json,
               args.prompt_out, args.prompt_text_out, args.raw_response_out)):
            parser.error("--all-splits requires --output-root and no single-split options")
        indices = pd.read_csv(args.dknn_csv)["split_index"].tolist()
        if not indices or len(indices) != len(set(indices)):
            parser.error("Batch GAP must contain nonempty unique split indices")
    else:
        if not args.output_json or args.responses_root or args.output_root:
            parser.error("Single split requires --output-json; --responses-root is batch-only")
        indices = [args.split_index]

    config = load_groups(args.groups_json) if args.groups_json else load_groups()
    if args.templates_only:
        names = expand_feature_groups(args.dataset_name, config)["feature_names"]
        feature_source = "template_preview_only"
    else:
        path = (Path(args.feature_names_npy) if args.feature_names_npy else
                load_dataset_spec(args.dataset_name).feature_names_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Actual feature-name file required: {path}. Use --feature-names-npy or "
                "--dry-run --templates-only for previews."
            )
        array = np.load(path, allow_pickle=True)
        if array.ndim != 1:
            raise ValueError("Feature-name array must be one-dimensional")
        names = list(map(str, array))
        feature_source = str(path.resolve())
    resolve_feature_groups(names, config, dataset=args.dataset_name)

    results = []
    for index in indices:
        gap = resolve_split_from_gap_csv(
            args.dknn_csv, parse_activity_list(args.seen), parse_activity_list(args.unseen), index
        )
        batch_dir = Path(args.output_root) / f"split_{gap['split_index']}" if args.all_splits else None
        responses = Path(args.responses_root) / f"split_{gap['split_index']}" if args.responses_root else None
        result = run_split(
            dataset=args.dataset_name,
            a1_path=args.a1_json,
            ontology_path=args.a1_ontology_json,
            gap_path=args.dknn_csv,
            gap_info=gap,
            output_path=batch_dir / "a2_attribute_plan.json" if batch_dir else args.output_json,
            feature_names=names,
            feature_source=feature_source,
            config=config,
            model=args.model,
            temperature=args.temperature,
            min_attributes=args.min_attributes,
            max_attributes=args.max_attributes,
            dry_run=args.dry_run,
            concepts_path=responses / "concepts.json" if responses else args.concepts_json,
            prompt_out=args.prompt_out,
            prompt_text_out=args.prompt_text_out,
            raw_response_out=args.raw_response_out,
            api_format=args.api_format,
            base_url=args.base_url,
            timeout=args.timeout,
            stream=args.stream,
        )
        results.append({"split_index": gap["split_index"], **result})
        print(json.dumps(results[-1], ensure_ascii=False))
    if args.all_splits:
        dump_json(results, Path(args.output_root) / "manifest.json")


if __name__ == "__main__":
    main()
