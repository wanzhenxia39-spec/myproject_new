#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt and LLM helper for gated top-k activity reranking.

This module separates the LLM reranking prompt into:
  1) a reusable system prompt containing the stable role, rules, and output format;
  2) a per-sample user prompt containing only the dynamic candidate set,
     original ranking and a compact local difference table.

The classifier script is responsible for deciding whether a sample is uncertain
enough to call the LLM. This module keeps the per-sample prompt compact: only
locally problematic attributes, the top feature evidence for those attributes,
and the needed feature-term glossary are included.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ORDINAL_LEVEL_TO_NUM = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
MISSING_VALUES = {"", "NA", "N/A", "None", "null", "nan"}
FULL_GLOSSARY = {
    "acc": "accelerometer",
    "gyro": "gyroscope",
    "x": "sensor x-axis",
    "y": "sensor y-axis",
    "z": "sensor z-axis",
    "mag": "vector magnitude, i.e. sqrt(x^2+y^2+z^2)",
    "norm": "vector magnitude, i.e. sqrt(x^2+y^2+z^2)",
    "mean": "average over the window",
    "std": "standard deviation",
    "iqr": "interquartile range (75th-25th percentile)",
    "max": "maximum value over the window",
    "median": "median value",
    "min": "minimum value",
    "peak": "maximum of the absolute sensor signal",
    "rms": "root mean square",
    "bp": "band power in a frequency band",
    "low": "frequency band roughly 0-0.5 Hz",
    "mid": "frequency band roughly 0.5-3 Hz",
    "high": "frequency band roughly larger than 3 Hz",
    "ratio": "(band power)/(total power)",
    "ent": "spectral entropy",
    "entropy": "spectral entropy",
    "centroid": "spectral centroid",
    "diff": "first-order difference",
    "zc_rate": "zero-crossing rate",
    "skew": "third statistical moment",
    "kurtosis": "fourth statistical moment",
}


def _is_missing_value(value: Any) -> bool:
    return str(value).strip() in MISSING_VALUES


def _closest_candidates_for_attribute(
    predicted_value: Any,
    candidate_values: Mapping[str, Any],
) -> Tuple[List[str], str]:
    """Return candidate(s) whose prototype value is closest to the predicted value.

    This is computed only from information already included in the reranking
    prompt: the sample's predicted attribute value and each candidate's
    prototype value for the same attribute.

    - For ordinal Low/Medium/High attributes, closeness is ordinal distance.
    - For nominal/categorical attributes, closeness is exact matching.
    """
    pred = str(predicted_value).strip()
    normalized_values = {act: str(value).strip() for act, value in candidate_values.items()}

    if _is_missing_value(pred):
        return [], "missing_predicted_value"

    non_missing_values = {
        act: value
        for act, value in normalized_values.items()
        if not _is_missing_value(value)
    }
    if not non_missing_values:
        return [], "missing_candidate_values"

    if pred in ORDINAL_LEVEL_TO_NUM and all(
        value in ORDINAL_LEVEL_TO_NUM for value in non_missing_values.values()
    ):
        pred_num = ORDINAL_LEVEL_TO_NUM[pred]
        distances = {
            act: abs(pred_num - ORDINAL_LEVEL_TO_NUM[value])
            for act, value in non_missing_values.items()
        }
        min_distance = min(distances.values())
        closest = [act for act, dist in distances.items() if dist == min_distance]
        if len(closest) == 1 and min_distance == 0.0:
            match_type = "unique_exact_ordinal_match"
        elif len(closest) > 1 and min_distance == 0.0:
            match_type = "tied_exact_ordinal_match"
        elif len(closest) == 1:
            match_type = "unique_nearest_ordinal_match"
        else:
            match_type = "tied_nearest_ordinal_match"
        return closest, match_type

    exact_matches = [act for act, value in non_missing_values.items() if value == pred]
    if len(exact_matches) == 1:
        return exact_matches, "unique_exact_nominal_match"
    if len(exact_matches) > 1:
        return exact_matches, "tied_exact_nominal_match"
    return [], "no_nominal_match"


def _attribute_definition_to_text(value: Any) -> str:
    """Return a short human-readable attribute meaning for compact prompts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        # Prefer semantically meaningful fields when an A2 plan is available.
        preferred_keys = [
            "meaning",
            "definition",
            "description",
            "semantic_meaning",
            "sensor_meaning",
            "observability",
            "sensor_observability",
            "rationale",
        ]
        parts: List[str] = []
        for key in preferred_keys:
            val = value.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        if parts:
            # Keep this compact because it is repeated in the local table.
            return "; ".join(parts[:2])
        # Avoid verbose metadata such as source/observed_levels in the prompt.
        return ""
    return ""


def build_local_difference_rows(
    candidate_acts: Sequence[str],
    pred_values: Mapping[str, str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    attribute_cols: Sequence[str],
    attribute_definitions: Optional[Any] = None,
    issue_attributes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Build compact rows for attributes whose prototype values differ inside the candidates.

    Each row contains only local evidence needed for reranking: attribute name,
    optional short meaning, the predicted sample value, each candidate prototype
    value, and a programmatically computed closest-candidate hint. The closest
    hint does not use the true label or any extra data; it is derived only from
    the predicted value and the candidate prototype values.
    """
    normalized_defs = infer_attribute_definitions(
        attribute_cols=attribute_cols,
        activity_vectors=activity_vectors,
        attribute_definitions=attribute_definitions,
    )
    selected_attrs = list(issue_attributes) if issue_attributes is not None else list(attribute_cols)
    rows: List[Dict[str, Any]] = []
    for attr in selected_attrs:
        candidate_values = {
            act: str(activity_vectors.get(act, {}).get(attr, "NA"))
            for act in candidate_acts
        }
        if len(set(candidate_values.values())) <= 1:
            continue

        predicted_value = str(pred_values.get(attr, "NA"))
        closest_candidates, match_type = _closest_candidates_for_attribute(
            predicted_value=predicted_value,
            candidate_values=candidate_values,
        )

        row: Dict[str, Any] = {
            "attribute": attr,
            "meaning": _attribute_definition_to_text(normalized_defs.get(attr)),
            "predicted": predicted_value,
            "closest_candidates": closest_candidates,
            "match_type": match_type,
        }
        row.update({act: value for act, value in candidate_values.items()})
        rows.append(row)
    return rows

def _looks_like_activity_vectors(value: Any, attribute_cols: Sequence[str]) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    first_value = next(iter(value.values()))
    if not isinstance(first_value, Mapping):
        return False
    attr_set = set(attribute_cols)
    return bool(attr_set.intersection(first_value.keys()))


def infer_attribute_definitions(
    attribute_cols: Sequence[str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    attribute_definitions: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Normalize optional attribute definitions for the prompt.

    Accepted inputs:
      - None: derive observed Low/Medium/High values from activity_vectors.
      - a2_attribute_plan.json content: {"attributes": [{"name": ..., ...}, ...]}.
      - a list of {"name": ..., ...} definition objects.
      - a mapping {attribute_name: definition}.
      - an activity-vector mapping; this is summarized into observed values.
    """
    attr_names = list(attribute_cols)

    if attribute_definitions is None or _looks_like_activity_vectors(attribute_definitions, attr_names):
        source_vectors = (
            attribute_definitions
            if _looks_like_activity_vectors(attribute_definitions, attr_names)
            else activity_vectors
        )
        definitions: Dict[str, Any] = {}
        for attr in attr_names:
            observed = sorted(
                {
                    str(vec[attr])
                    for vec in source_vectors.values()
                    if isinstance(vec, Mapping) and attr in vec
                }
            )
            definitions[attr] = {
                "source": "derived_from_activity_attribute_vectors",
                "observed_levels": observed,
            }
        return definitions

    if isinstance(attribute_definitions, Mapping) and isinstance(attribute_definitions.get("attributes"), list):
        return infer_attribute_definitions(
            attribute_cols=attr_names,
            activity_vectors=activity_vectors,
            attribute_definitions=attribute_definitions["attributes"],
        )

    if isinstance(attribute_definitions, Sequence) and not isinstance(attribute_definitions, (str, bytes)):
        by_name = {
            str(item.get("name")): item
            for item in attribute_definitions
            if isinstance(item, Mapping) and item.get("name") is not None
        }
        return {attr: by_name[attr] for attr in attr_names if attr in by_name}

    if isinstance(attribute_definitions, Mapping):
        return {
            attr: attribute_definitions[attr]
            for attr in attr_names
            if attr in attribute_definitions
        }

    return {}


def load_attribute_definitions_near_vectors(
    activity_vectors_path: str,
    activity_vectors: Mapping[str, Mapping[str, str]],
    attribute_cols: Sequence[str],
) -> Dict[str, Any]:
    """
    Load richer A2 definitions automatically when available.

    For pipeline runs, activity_attribute_vectors_a2.json usually sits next to
    a2_attribute_plan.json. If that plan is unavailable, derive lightweight
    definitions from the activity-vector JSON itself.
    """
    candidate_paths = []
    if activity_vectors_path:
        base_dir = os.path.dirname(activity_vectors_path)
        stem, _ = os.path.splitext(os.path.basename(activity_vectors_path))
        candidate_paths.extend(
            [
                os.path.join(base_dir, "a2_attribute_plan.json"),
                os.path.join(base_dir, f"{stem}.plan.json"),
            ]
        )

    for path in candidate_paths:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return infer_attribute_definitions(
                    attribute_cols=attribute_cols,
                    activity_vectors=activity_vectors,
                    attribute_definitions=json.load(f),
                )

    return infer_attribute_definitions(
        attribute_cols=attribute_cols,
        activity_vectors=activity_vectors,
        attribute_definitions=activity_vectors,
    )


def build_top3_rerank_system_prompt() -> str:
    """Stable system prompt for issue-attribute feature-evidence reranking."""
    return """You are an evidence-based reranker for wearable-IMU zero-shot human activity recognition.

Task:
Choose exactly one final activity from the provided allowed candidates.

Evidence:
- Issue attributes: only attributes where allowed candidates differ and the sample prediction conflicts with at least one candidate.
- Feature rows: top important IMU features for those issue-attribute classifiers, selected from seen-training/validation data.
- Glossary: short meanings for feature-name terms used in this prompt.

Rules:
- Use only the provided candidates, issue attributes, feature rows, and glossary.
- Percentile/z are relative to seen-training samples; feature rows do not give unseen-class distributions.
- Choose the candidate whose prototype is most consistent with the credible issue-attribute predictions and feature evidence.
- Return Low confidence when evidence is weak, tied, missing, or mostly follows original ranking.

Return JSON only:
{
  "chosen_activity": "one allowed candidate",
  "confidence": "Low/Medium/High",
  "original_top1_trust": "Low/Medium/High",
  "key_features": ["feature_1", "feature_2"],
  "key_attributes": ["attribute_1", "attribute_2"],
  "reason": "brief evidence-based explanation"
}""".strip()


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = [
        "| " + " | ".join(str(cell) if str(cell).strip() else "-" for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def _format_compact_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric != numeric:
        return "NA"
    return f"{numeric:.{digits}f}"


def _format_percentile(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric != numeric:
        return "NA"
    return f"{numeric:.1f}%"


def _as_list_of_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [item.strip() for item in text.split(",") if item.strip()]
        return [text]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _feature_terms(feature_name: Any) -> List[str]:
    text = str(feature_name).lower()
    raw_tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
    terms: List[str] = []
    for token in raw_tokens:
        if token in FULL_GLOSSARY:
            terms.append(token)
        if token.startswith("acc"):
            terms.append("acc")
        if token.startswith("gyro") or token.startswith("gyr"):
            terms.append("gyro")
        if token in {"norm", "magnitude"}:
            terms.append("mag")
        if token in {"fft", "freq", "frequency", "bp", "band"}:
            terms.append("bp" if token in {"bp", "band"} else token)
        if token in {"zerocrossing", "zero_crossing", "zc"}:
            terms.append("zc_rate")
    ordered: List[str] = []
    seen = set()
    for term in terms:
        if term in FULL_GLOSSARY and term not in seen:
            ordered.append(term)
            seen.add(term)
    return ordered


def build_needed_glossary(
    imu_feature_evidence_table: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, str]:
    needed: Dict[str, str] = {}
    for row in imu_feature_evidence_table or []:
        for field in ["feature_name", "modality", "axis", "feature_type"]:
            for term in _feature_terms(row.get(field, "")):
                needed[term] = FULL_GLOSSARY[term]
    return needed


def _format_glossary(glossary: Mapping[str, str]) -> str:
    if not glossary:
        return "NA"
    return "\n".join(f"- {term}: {meaning}" for term, meaning in sorted(glossary.items()))


def extract_attributes_from_feature_evidence(
    imu_feature_evidence_table: Optional[Sequence[Mapping[str, Any]]],
) -> List[str]:
    """Return unique issue attribute names referenced by feature evidence rows."""
    attrs: List[str] = []
    seen = set()
    for row in imu_feature_evidence_table or []:
        candidates = []
        target_attr = row.get("target_attribute")
        if target_attr is not None:
            candidates.append(str(target_attr))
        candidates.extend(_as_list_of_strings(row.get("selected_by_attributes")))
        for attr in candidates:
            if attr and attr not in seen:
                attrs.append(attr)
                seen.add(attr)
    return attrs


def build_candidate_attribute_patterns(
    candidate_names: Sequence[str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    attributes: Sequence[str],
    max_attributes: int = 12,
) -> Dict[str, Dict[str, str]]:
    """Build compact candidate prototype patterns for attributes referenced by evidence."""
    selected_attrs = [str(attr) for attr in attributes if str(attr).strip()][:max_attributes]
    patterns: Dict[str, Dict[str, str]] = {}
    for candidate in candidate_names:
        vec = activity_vectors.get(candidate, {})
        patterns[str(candidate)] = {attr: str(vec.get(attr, "NA")) for attr in selected_attrs}
    return patterns


def _format_candidate_attribute_patterns(
    patterns: Mapping[str, Mapping[str, str]],
) -> str:
    if not patterns:
        return "No candidate attribute patterns available."
    candidate_names = list(patterns.keys())
    attrs: List[str] = []
    for pattern in patterns.values():
        for attr in pattern.keys():
            if attr not in attrs:
                attrs.append(attr)
    if not attrs:
        return "No candidate attribute patterns available."
    rows = []
    for attr in attrs:
        rows.append([attr] + [patterns.get(candidate, {}).get(attr, "NA") for candidate in candidate_names])
    return _markdown_table(["Attribute"] + candidate_names, rows)


def _format_feature_evidence_table(
    imu_feature_evidence_table: Optional[Sequence[Mapping[str, Any]]],
    max_rows: int = 30,
) -> str:
    rows = list(imu_feature_evidence_table or [])[:max_rows]
    if not rows:
        return "No issue-attribute IMU feature evidence table is available."
    table_rows: List[List[Any]] = []
    for row in rows:
        target_attr = row.get("target_attribute") or ", ".join(_as_list_of_strings(row.get("selected_by_attributes"))) or "-"
        candidate_values = row.get("candidate_attribute_values", {})
        if isinstance(candidate_values, Mapping):
            candidate_text = "; ".join(f"{act}={val}" for act, val in candidate_values.items())
        else:
            candidate_text = str(candidate_values) if candidate_values else "-"
        closest = ", ".join(_as_list_of_strings(row.get("closest_candidates_by_attribute"))) or "-"
        sensor = row.get("sensor_position", "") or "unknown"
        signal_parts = [
            str(row.get("modality", "") or "").strip(),
            str(row.get("axis", "") or "").strip(),
            str(row.get("feature_type", "") or "").strip(),
        ]
        signal = "/".join([part for part in signal_parts if part]) or "-"
        importance = row.get("importance", row.get("aggregate_importance", row.get("importance_score", "NA")))
        table_rows.append(
            [
                target_attr,
                row.get("attribute_prediction", "NA"),
                candidate_text,
                closest,
                row.get("feature_name", ""),
                signal,
                _format_compact_float(importance, digits=4),
                _format_compact_float(row.get("validation_accuracy"), digits=3),
                _format_compact_float(row.get("test_value"), digits=4),
                _format_compact_float(row.get("z_score"), digits=2),
                _format_percentile(row.get("percentile")),
                row.get("observed_level", "NA"),
            ]
        )
    return _markdown_table(
        [
            "Issue attribute",
            "Predicted",
            "Candidate values",
            "Closest by attribute",
            "Feature",
            "Signal/type",
            "Importance",
            "Attr val acc",
            "Value",
            "z",
            "Percentile",
            "Level",
        ],
        table_rows,
    )


def summarize_feature_evidence(
    imu_feature_evidence_table: Optional[Sequence[Mapping[str, Any]]],
    top_n: int = 5,
) -> str:
    """Compact natural-language summary of the strongest feature evidence rows."""
    rows = list(imu_feature_evidence_table or [])
    if not rows:
        return "NA"

    def _importance(row: Mapping[str, Any]) -> float:
        try:
            return float(row.get("importance", row.get("aggregate_importance", row.get("importance_score", 0.0))))
        except Exception:
            return 0.0

    ranked = sorted(rows, key=_importance, reverse=True)[:top_n]
    parts: List[str] = []
    for row in ranked:
        feature = str(row.get("feature_name", "feature"))
        level = str(row.get("observed_level", "NA"))
        percentile = _format_percentile(row.get("percentile"))
        attrs = str(row.get("target_attribute") or ", ".join(_as_list_of_strings(row.get("selected_by_attributes"))) or "NA")
        parts.append(f"{feature}: {level} ({percentile}); issue attribute {attrs}")
    return "; ".join(parts)


def summarize_closest_support(
    local_rows: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """Summarize local closest-candidate evidence for reranking.

    unique_support counts attributes where the candidate is the only closest
    prototype to the predicted value. tied_support counts attributes where the
    candidate is among multiple closest candidates. These counts are derived
    only from predicted attribute values and candidate prototypes.
    """
    summary: Dict[str, Dict[str, int]] = {
        str(candidate): {"unique_support": 0, "tied_support": 0, "total_support": 0}
        for candidate in candidate_names
    }
    for row in local_rows:
        closest = [str(x) for x in row.get("closest_candidates", [])]
        if not closest:
            continue
        if len(closest) == 1:
            candidate = closest[0]
            if candidate in summary:
                summary[candidate]["unique_support"] += 1
                summary[candidate]["total_support"] += 1
        else:
            for candidate in closest:
                if candidate in summary:
                    summary[candidate]["tied_support"] += 1
                    summary[candidate]["total_support"] += 1
    return summary


def _format_support_summary(summary: Mapping[str, Mapping[str, int]]) -> str:
    parts: List[str] = []
    for candidate, counts in summary.items():
        parts.append(
            f"{candidate}: unique={int(counts.get('unique_support', 0))}, "
            f"tied={int(counts.get('tied_support', 0))}"
        )
    return "; ".join(parts) if parts else "NA"


def build_top3_rerank_user_prompt(
    top3_acts: Sequence[Tuple[str, float]],
    pred_values: Mapping[str, str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    attribute_cols: Sequence[str],
    dataset_name: str = "HAR dataset",
    attribute_definitions: Optional[Any] = None,
    imu_feature_evidence_table: Optional[Sequence[Mapping[str, Any]]] = None,
    sample_metadata: Optional[Mapping[str, Any]] = None,
    issue_attributes: Optional[Sequence[str]] = None,
) -> str:
    """Dynamic per-sample user prompt.

    The main prompt is candidate-aware: it first shows the local allowed-candidate
    attribute difference table, then attaches raw-feature evidence only for the
    issue attributes selected by the pipeline. This avoids feeding globally
    important but locally irrelevant feature rows to the LLM.
    """
    candidate_names = [act for act, _ in top3_acts]
    top1_top2_margin: Optional[float] = None
    if len(top3_acts) > 1:
        top1_top2_margin = float(top3_acts[0][1] - top3_acts[1][1])

    ranking_lines = [
        f"{rank}. {act}: {_format_float(score)}"
        for rank, (act, score) in enumerate(top3_acts, start=1)
    ]

    metadata = dict(sample_metadata or {})
    sample_lines = []
    for key in ["sample_id", "window_index", "dataframe_index", "split_index"]:
        if key in metadata:
            sample_lines.append(f"- {key}: {metadata[key]}")
    if metadata.get("evidence_candidate_scope"):
        sample_lines.append(f"- evidence_candidate_scope: {metadata['evidence_candidate_scope']}")
    if metadata.get("evidence_candidate_activities"):
        sample_lines.append(
            "- evidence_candidate_activities: "
            + ", ".join(_as_list_of_strings(metadata.get("evidence_candidate_activities")))
        )
    if metadata.get("llm_allowed_candidate_activities"):
        sample_lines.append(
            "- llm_allowed_candidate_activities: "
            + ", ".join(_as_list_of_strings(metadata.get("llm_allowed_candidate_activities")))
        )
    if "top2_top3_margin" in metadata:
        sample_lines.append(f"- top2_top3_margin: {metadata['top2_top3_margin']}")
    if "rerank_top3_evidence_margin_threshold" in metadata:
        sample_lines.append(
            "- rerank_top3_evidence_margin_threshold: "
            + str(metadata["rerank_top3_evidence_margin_threshold"])
        )
    if metadata.get("issue_attributes_for_feature_evidence"):
        sample_lines.append(
            "- issue_attributes_for_feature_evidence: "
            + ", ".join(_as_list_of_strings(metadata.get("issue_attributes_for_feature_evidence")))
        )
    sample_text = "\n".join(sample_lines) if sample_lines else "- sample metadata: NA"

    local_rows = build_local_difference_rows(
        candidate_acts=candidate_names,
        pred_values=pred_values,
        activity_vectors=activity_vectors,
        attribute_cols=attribute_cols,
        attribute_definitions=attribute_definitions,
        issue_attributes=issue_attributes,
    )
    support_summary = summarize_closest_support(local_rows, candidate_names)

    table_headers = ["Attribute", "Predicted"] + candidate_names + ["Closest"]
    table_rows: List[List[Any]] = []
    for row in local_rows:
        table_rows.append(
            [
                row.get("attribute", ""),
                row.get("predicted", "NA"),
                *[row.get(act, "NA") for act in candidate_names],
                ", ".join(row.get("closest_candidates", [])) or "-",
            ]
        )
    local_table_text = (
        _markdown_table(table_headers, table_rows)
        if table_rows
        else "No differing attributes among the allowed candidates."
    )

    evidence_attrs = extract_attributes_from_feature_evidence(imu_feature_evidence_table)
    if not evidence_attrs:
        evidence_attrs = [str(row.get("attribute")) for row in local_rows if row.get("attribute")]
    feature_table_text = _format_feature_evidence_table(imu_feature_evidence_table)
    strongest_feature_summary = summarize_feature_evidence(imu_feature_evidence_table)
    glossary_text = _format_glossary(build_needed_glossary(imu_feature_evidence_table))

    return f"""Dataset: {dataset_name}

Sample metadata:
{sample_text}

Allowed candidates ranked by original attribute-cosine score:
{os.linesep.join(ranking_lines)}

Original Top-1: {candidate_names[0] if candidate_names else "NA"}
Top1-Top2 margin: {_format_float(top1_top2_margin)}

Closest-support summary based on local difference attributes:
{_format_support_summary(support_summary)}

Local attribute difference table:
{local_table_text}

Feature-term glossary:
{glossary_text}

Issue-attribute feature evidence summary:
{strongest_feature_summary}

Issue-attribute IMU feature evidence table:
{feature_table_text}

Use the local attribute difference table to identify the attributes that distinguish the allowed candidates. If evidence_candidate_scope is top1_top2_only, rank-3 was excluded before this prompt because its cosine score was far below rank-2; do not infer or choose any omitted activity. Use the issue-attribute feature evidence only to interpret whether the sample's predicted value on those attributes is credible. Use commonsense only for interpreting provided feature terms and activity prototypes. Choose one final activity from the allowed candidates. Return JSON only.""".strip()


def build_top3_rerank_prompt(
    top3_acts: Sequence[Tuple[str, float]],
    pred_values: Mapping[str, str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    attribute_cols: Sequence[str],
    dataset_name: str = "HAR dataset",
    attribute_definitions: Optional[Any] = None,
    imu_feature_evidence_table: Optional[Sequence[Mapping[str, Any]]] = None,
    sample_metadata: Optional[Mapping[str, Any]] = None,
    issue_attributes: Optional[Sequence[str]] = None,
) -> str:
    """Backward-compatible combined prompt for logging or older callers."""
    return (
        "[SYSTEM PROMPT]\n"
        + build_top3_rerank_system_prompt()
        + "\n\n[USER PROMPT]\n"
        + build_top3_rerank_user_prompt(
            top3_acts=top3_acts,
            pred_values=pred_values,
            activity_vectors=activity_vectors,
            attribute_cols=attribute_cols,
            dataset_name=dataset_name,
            attribute_definitions=attribute_definitions,
            imu_feature_evidence_table=imu_feature_evidence_table,
            sample_metadata=sample_metadata,
            issue_attributes=issue_attributes,
        )
    )


def call_llm(system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-genai is required for --candidate_rerank llm_top3.") from exc

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY or GEMINI_API_KEY before using --candidate_rerank llm_top3.")

    base_url = os.environ.get("LLM_BASE_URL")
    trust_env = os.environ.get("LLM_TRUST_ENV", "").strip().lower() in {"1", "true", "yes"}
    http_options = {
        "client_args": {"trust_env": trust_env},
        "async_client_args": {"trust_env": trust_env},
    }
    if base_url:
        http_options["base_url"] = base_url

    client = genai.Client(api_key=api_key, http_options=http_options)
    config_kwargs = {"temperature": temperature}
    # google-genai supports system_instruction for Gemini models. If a custom
    # compatible endpoint does not support it, the exception will surface clearly.
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=user_prompt)])],
        config=types.GenerateContentConfig(**config_kwargs),
    )
    if getattr(resp, "text", None):
        return resp.text
    parts = []
    for cand in getattr(resp, "candidates", []) or []:
        for part in getattr(cand.content, "parts", []) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "\n".join(parts).strip()


def _normalize_level(value: Any, default: str = "Low") -> str:
    level = str(value).strip()
    return level if level in {"Low", "Medium", "High"} else default


def parse_llm_choice(response_text: str, allowed_candidates: Sequence[str]) -> Dict[str, Any]:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    choice = str(parsed.get("chosen_activity", "")).strip()
    if choice not in set(allowed_candidates):
        raise ValueError(
            f"LLM chose [{choice}], which is not in allowed candidates: {list(allowed_candidates)}"
        )

    key_attributes = parsed.get("key_attributes", [])
    if isinstance(key_attributes, str):
        key_attributes = [key_attributes]
    elif not isinstance(key_attributes, list):
        key_attributes = []

    key_features = parsed.get("key_features", [])
    if isinstance(key_features, str):
        key_features = [key_features]
    elif not isinstance(key_features, list):
        key_features = []

    return {
        "chosen_activity": choice,
        "confidence": _normalize_level(parsed.get("confidence", "Low"), default="Low"),
        "original_top1_trust": _normalize_level(parsed.get("original_top1_trust", "Medium"), default="Medium"),
        "key_features": [str(x) for x in key_features],
        "key_attributes": [str(x) for x in key_attributes],
        "reason": str(parsed.get("reason", "")).strip(),
    }


def accept_llm_change(
    policy: str,
    confidence: str,
    original_top1_trust: str,
    llm_choice: str,
    original_top1: str,
    support_summary: Mapping[str, Mapping[str, int]],
) -> bool:
    """Optional rule guard for deciding whether to accept a non-Top1 LLM choice.

    The prompt can be made less conservative so that the LLM proposes more
    candidate changes. This guard prevents overly aggressive changes when the
    LLM confidence/support is weak. The classifier script may use this helper
    when it wants controlled acceptance instead of blindly using chosen_activity.
    """
    if llm_choice == original_top1:
        return False

    confidence = _normalize_level(confidence, default="Low")
    original_top1_trust = _normalize_level(original_top1_trust, default="Medium")
    llm_counts = support_summary.get(llm_choice, {})
    top1_counts = support_summary.get(original_top1, {})
    llm_unique = int(llm_counts.get("unique_support", 0))
    top1_unique = int(top1_counts.get("unique_support", 0))
    llm_total = int(llm_counts.get("total_support", 0))
    top1_total = int(top1_counts.get("total_support", 0))

    if policy == "none":
        return True
    if policy == "conservative":
        return confidence == "High" and llm_unique > top1_unique and original_top1_trust != "High"
    if policy == "balanced":
        return (
            confidence == "High"
            or (
                confidence == "Medium"
                and original_top1_trust != "High"
                and (llm_unique > top1_unique or llm_total >= top1_total)
            )
        )
    if policy == "aggressive":
        return confidence in {"Medium", "High"} and original_top1_trust != "High"

    raise ValueError(f"Unknown LLM acceptance policy: {policy}")


def select_top3_activity_with_llm(
    top3_acts: Sequence[Tuple[str, float]],
    pred_values: Mapping[str, str],
    activity_vectors: Mapping[str, Mapping[str, str]],
    attribute_cols: Sequence[str],
    model: str,
    temperature: float,
    dataset_name: str = "HAR dataset",
    attribute_definitions: Optional[Any] = None,
    acceptance_policy: str = "none",
    imu_feature_evidence_table: Optional[Sequence[Mapping[str, Any]]] = None,
    sample_metadata: Optional[Mapping[str, Any]] = None,
    issue_attributes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Select a Top-3 activity using the LLM reranker.

    acceptance_policy controls an optional post-LLM guard:
      - none: expose the raw LLM chosen_activity as final_activity.
      - conservative/balanced/aggressive: compute whether a non-Top1 LLM choice
        should be accepted according to confidence, original_top1_trust, and
        closest-support evidence.

    Existing callers can ignore final_activity and use chosen_activity directly.
    New callers should prefer final_activity when using an acceptance policy.
    """
    system_prompt = build_top3_rerank_system_prompt()
    user_prompt = build_top3_rerank_user_prompt(
        top3_acts=top3_acts,
        pred_values=pred_values,
        activity_vectors=activity_vectors,
        attribute_cols=attribute_cols,
        dataset_name=dataset_name,
        attribute_definitions=attribute_definitions,
        imu_feature_evidence_table=imu_feature_evidence_table,
        sample_metadata=sample_metadata,
        issue_attributes=issue_attributes,
    )
    response_text = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
    )
    parsed = parse_llm_choice(response_text, [act for act, _ in top3_acts])
    candidate_names = [act for act, _ in top3_acts]
    original_top1 = candidate_names[0] if candidate_names else ""
    local_rows = build_local_difference_rows(
        candidate_acts=candidate_names,
        pred_values=pred_values,
        activity_vectors=activity_vectors,
        attribute_cols=attribute_cols,
        attribute_definitions=attribute_definitions,
        issue_attributes=issue_attributes,
    )
    support_summary = summarize_closest_support(local_rows, candidate_names)
    evidence_attrs = extract_attributes_from_feature_evidence(imu_feature_evidence_table)
    if not evidence_attrs:
        evidence_attrs = [str(row.get("attribute")) for row in local_rows if row.get("attribute")]
    candidate_attribute_patterns = build_candidate_attribute_patterns(
        candidate_names=candidate_names,
        activity_vectors=activity_vectors,
        attributes=evidence_attrs,
    ) if evidence_attrs else {}
    should_change_original_top1 = parsed["chosen_activity"] != original_top1
    accepted_change = accept_llm_change(
        policy=acceptance_policy,
        confidence=parsed["confidence"],
        original_top1_trust=parsed["original_top1_trust"],
        llm_choice=parsed["chosen_activity"],
        original_top1=original_top1,
        support_summary=support_summary,
    ) if should_change_original_top1 else False
    final_activity = parsed["chosen_activity"] if (not should_change_original_top1 or accepted_change) else original_top1

    return {
        **parsed,
        "final_activity": final_activity,
        "original_top1": original_top1,
        "should_change_original_top1": bool(should_change_original_top1),
        "accepted_change": bool(accepted_change),
        "acceptance_policy": acceptance_policy,
        "closest_support_summary": support_summary,
        "imu_feature_evidence_table": list(imu_feature_evidence_table or []),
        "candidate_attribute_patterns": candidate_attribute_patterns,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt": "[SYSTEM PROMPT]\n" + system_prompt + "\n\n[USER PROMPT]\n" + user_prompt,
        "raw_response": response_text,
        "local_difference_table": local_rows,
        "feature_term_glossary": build_needed_glossary(imu_feature_evidence_table),
    }


def build_example_prompt() -> str:
    attribute_cols = [
        "motion_intensity",
        "cadence",
        "smoothness",
        "posture_inclination",
        "dominant_body_region",
        "speed",
    ]
    top3_acts = [
        ("walking", 0.8421),
        ("running", 0.8175),
        ("cycling", 0.7812),
    ]
    pred_values = {
        "motion_intensity": "High",
        "cadence": "High",
        "smoothness": "Medium",
        "posture_inclination": "Medium",
        "dominant_body_region": "Lower body",
        "speed": "High",
    }
    activity_vectors = {
        "walking": {
            "motion_intensity": "Medium",
            "cadence": "Medium",
            "smoothness": "High",
            "posture_inclination": "Medium",
            "dominant_body_region": "Lower body",
            "speed": "Medium",
        },
        "running": {
            "motion_intensity": "High",
            "cadence": "High",
            "smoothness": "Medium",
            "posture_inclination": "Medium",
            "dominant_body_region": "Lower body",
            "speed": "High",
        },
        "cycling": {
            "motion_intensity": "Medium",
            "cadence": "High",
            "smoothness": "High",
            "posture_inclination": "Low",
            "dominant_body_region": "Lower body",
            "speed": "Medium",
        },
    }
    imu_feature_evidence_table = [
        {
            "feature_name": "ankle_acc_norm_rms",
            "sensor_position": "ankle",
            "modality": "accelerometer",
            "axis": "norm",
            "feature_type": "dynamic_intensity",
            "physical_meaning": "overall lower-limb acceleration magnitude",
            "selected_by_attributes": ["motion_intensity", "speed"],
            "aggregate_importance": 0.182,
            "test_value": 2.81,
            "z_score": 1.36,
            "percentile": 88.4,
            "observed_level": "High",
            "evidence_interpretation": "Strong lower-limb acceleration amplitude.",
        },
        {
            "feature_name": "ankle_gyro_z_fft_dom_freq",
            "sensor_position": "ankle",
            "modality": "gyroscope",
            "axis": "z",
            "feature_type": "frequency",
            "physical_meaning": "dominant ankle rotation rhythm",
            "selected_by_attributes": ["cadence"],
            "aggregate_importance": 0.136,
            "test_value": 1.72,
            "z_score": 0.94,
            "percentile": 74.1,
            "observed_level": "High",
            "evidence_interpretation": "Relatively high periodic ankle rotation cadence.",
        },
    ]
    return build_top3_rerank_prompt(
        top3_acts=top3_acts,
        pred_values=pred_values,
        activity_vectors=activity_vectors,
        attribute_cols=attribute_cols,
        dataset_name="PAMAP2",
        attribute_definitions=activity_vectors,
        imu_feature_evidence_table=imu_feature_evidence_table,
        sample_metadata={"sample_id": 1234},
        issue_attributes=["motion_intensity", "cadence", "smoothness", "speed"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print an example system/user prompt used by the LLM top-3 reranker."
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the example prompt. If omitted, print to stdout.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prompt = build_example_prompt()
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(prompt)
            f.write("\n")
        print(f"Example prompt saved to: {args.output}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
