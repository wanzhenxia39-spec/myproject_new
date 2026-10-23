"""Validate conceptual A1 operations separately from A2 feature bindings."""
from __future__ import annotations

from collections import Counter
import re

from zshar.features.catalog import resolve_plan_features
from zshar.features.groups import load_groups

SEMANTIC_FIELDS = ("name", "definition", "levels")
BINDING_FIELDS = ("primary_groups", "auxiliary_groups", "group_evidence", "feature_selectors")


def semantic_attributes(plan):
    """Only concept semantics reach vector completion; selectors are feature-side."""
    return [{key: attr[key] for key in SEMANTIC_FIELDS} for attr in plan["attributes"]]


def validate_concepts(plan, *, a1_names, seen, unseen, dataset, split_index,
                      min_attributes=1, max_attributes=25, source_only=False, allow_free_edits=False):
    if not isinstance(plan, dict) or plan.get("schema_version") != "2.0":
        raise ValueError("Expected refinement schema_version 2.0")
    if plan.get("dataset") != dataset or plan.get("split_index") != split_index:
        raise ValueError("Refinement dataset/split does not match input")
    attrs = plan.get("attributes")
    if not isinstance(attrs, list) or not min_attributes <= len(attrs) <= max_attributes:
        raise ValueError("A2 attribute count outside allowed bounds")
    names, definitions = [], []
    forbidden = {"acc", "gyro", "accelerometer", "gyroscope", "magnetometer", "x", "y", "z",
                 "hand", "ankle", "chest", "hip", "arm", "leg", "torso", "phone", "watch", "wrist"}
    for attr in attrs:
        if not isinstance(attr, dict):
            raise ValueError("Each A2 attribute must be an object")
        if set(attr) - set(SEMANTIC_FIELDS + BINDING_FIELDS):
            raise ValueError("Unexpected attribute fields; feature restrictions belong in feature_selectors")
        name = attr.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name):
            raise ValueError("A2 names must be snake_case physical concepts")
        if set(name.split("_")) & forbidden or re.search(r"(?:^|_)g\d+(?:_|$)", name):
            raise ValueError(f"Feature-side position/modality/axis/group in attribute name: {name}")
        if any(key in attr for key in ("body_position", "sensor_type", "axes", "imu_observability")):
            raise ValueError("Position/modality/axis restrictions belong in feature_selectors")
        definition = attr.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError("Each physical concept needs a definition")
        levels = attr.get("levels")
        if not isinstance(levels, dict) or set(levels) != {"Low", "Medium", "High"} or any(
            not isinstance(v, str) or not v.strip() for v in levels.values()
        ):
            raise ValueError("Each A2 attribute must define Low/Medium/High")
        names.append(name)
        definitions.append(" ".join(definition.lower().split()))
    if len(set(names)) != len(names) or len(set(definitions)) != len(definitions):
        raise ValueError("Duplicate A2 names or identical definitions")
    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("Record concept-lineage decisions for every A1 input and A2 output")
    inputs, outputs = [], []
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("Each operation must be an object")
        action, sources, targets = op.get("action"), op.get("inputs"), op.get("outputs")
        if not isinstance(sources, list) or not isinstance(targets, list):
            raise ValueError("Operation inputs/outputs must be lists")
        if any(not isinstance(s, str) or s not in a1_names for s in sources) or any(
            not isinstance(t, str) or t not in names for t in targets
        ):
            raise ValueError("Unknown operation input/output")
        valid = ((action == "keep" and len(sources) == 1 and targets == sources) or
                 (action == "split" and len(sources) == 1 and len(targets) >= 2) or
                 (action == "merge" and len(sources) >= 2 and len(targets) == 1) or
                 (action == "drop" and len(sources) == 1 and not targets) or
                 ((source_only or allow_free_edits) and action == "add" and not sources and len(targets) == 1) or
                 (allow_free_edits and action == "replace" and len(sources) == len(targets) == 1
                  and sources != targets) or
                 (source_only and action == "replace" and sources and targets and
                  not (len(sources) == len(targets) == 1 and sources == targets)))
        if not valid:
            raise ValueError("Invalid concept-lineage action or cardinality")
        if not isinstance(op.get("reason"), str) or not op["reason"].strip():
            raise ValueError("Each operation needs a reason")
        for field, allowed in (("target_unseen", unseen), ("contrast_unseen", unseen), ("seen_anchors", seen)):
            values = op.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values):
                raise ValueError(f"Invalid {field} in operation")
        if action in {"split", "merge"} and not source_only and not allow_free_edits:
            if not op.get("target_unseen") or not op.get("seen_anchors"):
                raise ValueError("Changed concepts need target unseen classes and seen anchors")
            if len(unseen) > 1 and (not op.get("contrast_unseen") or
                                   not set(op["contrast_unseen"]) - set(op["target_unseen"])):
                raise ValueError("Changed concepts need another unseen class to distinguish")
        inputs.extend(sources)
        outputs.extend(targets)
    if Counter(inputs) != Counter(a1_names) or Counter(outputs) != Counter(names):
        raise ValueError("Operations must account for every A1 input and A2 output exactly once")
    changed = any(op["action"] != "keep" for op in operations)
    if type(plan.get("refinement_needed")) is not bool or plan["refinement_needed"] != changed:
        raise ValueError("refinement_needed must agree with conceptual operations")
    for key in ("decision_reason", "semantic_redundancy_review"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"Missing {key}")


def bind_features(concepts, binding_response):
    if not isinstance(binding_response, dict) or set(binding_response) != {"attribute_bindings"}:
        raise ValueError("Binding response must contain only attribute_bindings")
    bindings = binding_response["attribute_bindings"]
    if not isinstance(bindings, list) or any(not isinstance(b, dict) for b in bindings):
        raise ValueError("attribute_bindings must be a list of objects")
    names = [a["name"] for a in concepts["attributes"]]
    if any(not isinstance(b.get("attribute"), str) for b in bindings):
        raise ValueError("Each binding needs a string attribute name")
    if Counter(b.get("attribute") for b in bindings) != Counter(names):
        raise ValueError("Bind each A2 attribute exactly once")
    allowed = {"attribute", "primary_groups", "auxiliary_groups", "group_evidence", "feature_selectors"}
    if any(set(b) != allowed for b in bindings):
        raise ValueError("Binding stage cannot modify concept semantics or omit binding fields")
    by_name = {b["attribute"]: b for b in bindings}
    return {**concepts, "attributes": [
        {**a, **{k: v for k, v in by_name[a["name"]].items() if k != "attribute"}}
        for a in concepts["attributes"]]}


def validate_bound_plan(plan, *, feature_names=None, groups_config=None, **context):
    validate_concepts(plan, **context)
    config = groups_config or load_groups()
    if feature_names is None:
        raise ValueError("Actual feature names are required to validate A2 bindings")
    return resolve_plan_features(plan, feature_names, dataset=context["dataset"], config=config)
