#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset-independent, split-specific two-stage attribute refinement.

Inputs:
- A fixed seen/unseen split.
- Initial activity attribute vectors A1.
- D_kNN result for the split, including per-unseen values and nearest seen classes.

Output:
- A validated A2 schema, operation audit, and exact per-attribute feature mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import numpy as np

from zshar.datasets import load_dataset_spec
from zshar.features.groups import load_groups, expand_feature_groups, resolve_feature_groups
from zshar.features.catalog import feature_catalog
from zshar.semantics.refinement import (validate_concepts, bind_features,
                                        validate_bound_plan, semantic_attributes)

DEFAULT_MAX_ATTRIBUTES_LIMIT = 25


def parse_activity_list(text: str) -> List[str]:
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


def write_text(text: str, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def default_metadata_path(dataset_name: str) -> Optional[Path]:
    """Resolve metadata through the shared dataset registry."""
    try:
        from zshar.datasets import load_dataset_spec

        return load_dataset_spec(dataset_name).metadata_path
    except (ValueError, FileNotFoundError):
        return None


def load_dataset_metadata(dataset_name: str, metadata_json: str) -> Dict[str, Any]:
    if metadata_json:
        return load_json(metadata_json)
    path = default_metadata_path(dataset_name)
    if path and path.exists():
        return load_json(str(path))
    return {
        "dataset_name": dataset_name,
        "sampling_rate_hz": None,
        "sensor_units": [],
        "note": "No dataset metadata JSON was provided. The planner must avoid assuming unavailable sensors, body positions, or axes.",
    }


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


def call_gemini(prompt: str, model: str, temperature: float) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for live LLM calls. Install it or use --dry-run."
        ) from exc

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY or GEMINI_API_KEY before live LLM calls.")

    base_url = os.environ.get("LLM_BASE_URL")
    trust_env = os.environ.get("LLM_TRUST_ENV", "").strip().lower() in {"1", "true", "yes"}
    http_options = {
        "client_args": {"trust_env": trust_env},
        "async_client_args": {"trust_env": trust_env},
    }
    if base_url:
        http_options["base_url"] = base_url
    client = genai.Client(api_key=api_key, http_options=http_options)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(temperature=temperature),
    )
    if getattr(resp, "text", None):
        return resp.text
    parts = []
    for cand in getattr(resp, "candidates", []) or []:
        for part in getattr(cand.content, "parts", []) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "\n".join(parts).strip()


def attribute_names(a1_vectors: Mapping[str, Mapping[str, str]]) -> List[str]:
    if not isinstance(a1_vectors, dict) or not a1_vectors:
        raise ValueError("A1 attribute JSON is empty.")
    first = next(iter(a1_vectors.values()))
    if not isinstance(first, dict) or not first:
        raise ValueError('A1 class vectors must be nonempty attribute dictionaries')
    return list(first)


def resolve_split_from_gap_csv(gap_csv, seen, unseen, split_index):
    """Select exactly one split; explicit classes must agree with its GAP row."""
    frame = pd.read_csv(gap_csv)
    if frame['split_index'].duplicated().any():
        raise ValueError('GAP split_index must be unique')
    if split_index is not None:
        matched = frame[frame['split_index'] == int(split_index)]
    elif unseen:
        matched = frame[frame['unseen_classes'].map(
            lambda value: set(parse_activity_list(str(value))) == set(unseen))]
    else:
        raise ValueError('Specify --split-index, --unseen, or --all-splits')
    if len(matched) != 1:
        raise ValueError('Could not locate exactly one requested split in --dknn-csv')
    row = matched.iloc[0].to_dict()
    row_seen = parse_activity_list(str(row['seen_classes']))
    row_unseen = parse_activity_list(str(row['unseen_classes']))
    if ((seen and set(seen) != set(row_seen)) or
            (unseen and set(unseen) != set(row_unseen))):
        raise ValueError('Explicit seen/unseen classes disagree with GAP split')
    if (not row_seen or not row_unseen or set(row_seen) & set(row_unseen) or
            len(set(row_seen)) != len(row_seen) or len(set(row_unseen)) != len(row_unseen)):
        raise ValueError('GAP split must contain nonempty, unique, disjoint class lists')
    per_unseen = json.loads(row['D_kNN_per_unseen_json'])
    if not isinstance(per_unseen, dict) or set(per_unseen) != set(row_unseen):
        raise ValueError('D_kNN_per_unseen must cover precisely this split unseen classes')
    values = [row['D_kNN'], *per_unseen.values()]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('D_kNN values must be finite nonnegative numbers')
    neighbors = json.loads(row.get('nearest_seen_topk_json', '{}'))
    if not isinstance(neighbors, dict) or (neighbors and set(neighbors) != set(row_unseen)):
        raise ValueError('Nearest-seen diagnostics must refer to this split unseen classes')
    for pairs in neighbors.values():
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list) or len(pair) != 2 or pair[0] not in row_seen or
            type(pair[1]) not in (int, float) or not math.isfinite(pair[1]) or pair[1] < 0
            for pair in pairs
        ):
            raise ValueError('Invalid nearest-seen class/distance diagnostics')
    return {'split_index': int(row['split_index']), 'seen': row_seen, 'unseen': row_unseen,
            'D_kNN': row['D_kNN'], 'D_kNN_per_unseen': per_unseen,
            'nearest_seen_topk': neighbors,
            'a1_sha256': row.get('a1_sha256') if isinstance(row.get('a1_sha256'), str) else None}


def format_gap_info_for_prompt(gap_info):
    return json.dumps({key: gap_info[key] for key in (
        'split_index', 'seen', 'unseen', 'D_kNN', 'D_kNN_per_unseen', 'nearest_seen_topk'
    ) if key in gap_info}, ensure_ascii=False, indent=2)


def group_context(dataset, feature_names, config, *, include_columns=True):
    partition = resolve_feature_groups(feature_names, config, dataset=dataset)
    return {gid: {**group, 'available_count': len(partition['group_to_features'][gid]),
                  **({'available_features': partition['group_to_features'][gid]} if include_columns else {})}
            for gid, group in config['groups'].items()}


def template_tags(dataset, feature_names, config):
    """Avoid repeating identical tag definitions for every sensor placement."""
    tags = feature_catalog(feature_names, dataset=dataset, config=config)
    return {'{prefix}' + name[len(info['position']):]: {
        'modalities': info['modalities'], 'axes': info['axes']}
        for name, info in tags.items()}


def build_prompt(dataset_name, seen, unseen, a1_vectors, gap_info, dataset_metadata,
                 min_new_attributes=None, max_new_attributes=None,
                 min_attributes=1, max_attributes=25, *, feature_groups=None):
    """Concept stage. Legacy new-count arguments are accepted but impose no quota."""
    return f'''You design transferable ordinal physical-motion attributes for IMU zero-shot HAR.
Work independently on the specified split. Use class semantics and the supplied diagnostics;
never use unseen recordings, sample statistics, test labels, or recognition performance.

1. Context
Dataset: {dataset_name}
Seen activities: {json.dumps(seen, ensure_ascii=False)}
Unseen activities: {json.dumps(unseen, ensure_ascii=False)}
Source attribute names (for operation lineage): {json.dumps(attribute_names(a1_vectors))}
Class-level source prototypes (diagnostic input, not sensor samples):
{json.dumps(a1_vectors, ensure_ascii=False, indent=2)}
Dataset metadata (observability constraints only):
{json.dumps(dataset_metadata, ensure_ascii=False, indent=2)}
Fixed feature groups (feasibility reference; binding is a separate later step):
{json.dumps(feature_groups or {}, ensure_ascii=False, indent=2)}

2. Split-specific D_kNN objective
{format_gap_info_for_prompt(gap_info)}
Larger D_kNN means poorer local coverage of an unseen class by seen classes in the
source semantic space. Prioritize the largest per-unseen gaps, inspect nearest seen
anchors and source prototypes, and explain what physically meaningful distinctions
can distinguish each difficult unseen class from OTHER UNSEEN classes while retaining
transferable levels supported by seen activities. D_kNN guides refinement, not activity
naming or a demand to add dimensions. A high gap alone does not prove ambiguity.
Do not lower D_kNN by collapsing concepts or erasing unseen distinctions. Raw distances
across changed dimensions do not by themselves establish improvement. Do not claim
an achieved A2 D_kNN reduction before A2 prototypes are generated and measured.

3. Compact refinement and semantic review
- First decide whether changes are justified. If no useful, observable, seen-supported
  distinctions exist, keep the source concepts and set refinement_needed=false.
- Record every source exactly once with keep (one source -> same name), split (one
  source -> two or more distinct physical concepts), merge (two or more sources ->
  one physical concept), or drop (one source -> no output). Account for every output
  exactly once. Use no untracked additions, renamings, or one-to-one replacements.
- A split must state the different physical factors and added unseen discrimination;
  a merge must explain the common physical meaning and why useful distinctions survive;
  a drop must explain redundancy, lack of observability/seen support, or task irrelevance.
  Each split/merge must name target_unseen, contrast_unseen (another unseen class when
  present), seen_anchors and a concise reason connecting them to D_kNN and transfer.
- Each attribute describes an abstract human-motion physical concept. Do not refine
  by body position, sensor type, axis, activity identity or environment. These restrictions
  apply to names, definitions AND level descriptions. Positions/modalities/axes belong
  only to feature-side selection in the later binding step. Distinguish average posture
  inclination from posture stability/variability, and sustained change from isolated impact.
- Reject synonyms, mere renamings and concepts with no new physical meaning. Statistical
  correlation or similar Low/Medium/High patterns alone is NOT semantic redundancy:
  distinct correlated motion factors may be retained for a later classifier chain.
  Review each concept against the others; give a concise semantic_redundancy_review
  explaining closest conceptual overlaps and any retained dependencies.
- Require plausible evidence in the fixed available IMU feature groups. Do not invent
  a feature group or demand unavailable measurements. Feasibility is not proof of
  predictability; learnability must later be evaluated on seen data.
- Return {min_attributes} to {max_attributes} attributes; impose no minimum additions.
  Each needs a nonempty definition and exactly Low, Medium, High with increasing degree.
  Kept concepts retain their physical meaning; do not silently redefine them.

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
output attribute separately when an operation has multiple outputs. Cite only available
feature groups as feasibility evidence; actual feature binding remains a later step.

Output JSON only, with this structure (replace illustrative placeholders):
{{"schema_version":"2.0", "dataset":{json.dumps(dataset_name)},
 "split_index":{json.dumps(gap_info['split_index'])}, "refinement_needed":true,
 "decision_reason":"why this split needs these changes or why keep is sufficient",
 "attributes":[{{"name":"physical_concept", "definition":"abstract physical meaning",
   "levels":{{"Low":"low degree", "Medium":"intermediate degree", "High":"high degree"}}}}],
 "operations":[{{"action":"keep|split|merge|drop", "inputs":["source_name"],
   "outputs":["output_name"], "reason":"physical distinction, gap evidence and transfer rationale; per output: supporting groups, expected sensor evidence, prediction ambiguity",
   "target_unseen":[], "contrast_unseen":[], "seen_anchors":[]}}],
 "semantic_redundancy_review":"conceptual overlaps reviewed; statistical dependence is allowed"}}
Do not output feature bindings, selectors, extra attribute fields, or reasoning traces.
'''.strip()


def build_binding_prompt(concepts, feature_names, config, metadata):
    dataset = concepts['dataset']
    return f'''Bind fixed IMU evidence to EVERY finalized A2 physical concept below.
This is a separate review after conceptual refinement. Reconsider all mappings from
scratch, including kept source attributes. Earlier A1 group associations are not binding
and are intentionally not supplied. Do not change concept names, definitions or levels.

Frozen concepts:
{json.dumps(semantic_attributes(concepts), ensure_ascii=False, indent=2)}
Dataset: {dataset}; split_index: {concepts['split_index']}
Dataset metadata: {json.dumps(metadata, ensure_ascii=False)}
Fixed G1-G14 catalog and ACTUALLY AVAILABLE columns:
{json.dumps(group_context(dataset, feature_names, config), ensure_ascii=False, indent=2)}
Structured tags shared by each column template (prefix encodes position):
{json.dumps(template_tags(dataset, feature_names, config), ensure_ascii=False)}
Allowed positions and their physical meaning:
{json.dumps(config['dataset_sensors'][dataset], ensure_ascii=False)}

For each attribute choose at least one primary_groups entry from G1-G14. auxiliary_groups
may be empty. For every selected group give group_evidence explaining how its actual
statistics reflect the concept and ordinal levels, including limitations or indirect
proxies. Group names alone do not establish physical support; examine the feature list.
Never select a group with no available columns. No new feature extraction is allowed.
Position, modality and axis are optional FEATURE-SIDE filters, never semantic refinements.
Use feature_selectors with positions (exact configured prefixes), modalities (acc/gyro),
axes (x/y/z). Empty lists mean unrestricted. Derived features (magnitude, gravity angle,
EVA, CAGH, AVG/AVH, acceleration-rotation ratios) may depend on several axes/modalities:
ALL required tags must satisfy a filter, as listed above. At least one column in EACH
selected primary or auxiliary group must survive filters. Prefer unrestricted selectors
unless there is a physical reason. Do not turn a general concept into a placement-specific
concept. Do not infer empirical predictability or inspect unseen recordings.

Return JSON only. Bind each frozen name exactly once; use precisely these binding keys:
{{"attribute_bindings":[{{"attribute":"exact_frozen_name", "primary_groups":["G3"],
 "auxiliary_groups":[], "group_evidence":{{"G3":"specific physical evidence and limitations"}},
 "feature_selectors":{{"positions":[], "modalities":[], "axes":[]}}}}]}}
'''.strip()


def validate_plan(plan, min_attributes=1, max_attributes=25, **context):
    """Validate the full schema with trusted split/source context and actual columns."""
    return validate_bound_plan(plan, min_attributes=min_attributes,
                               max_attributes=max_attributes, **context)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_split(*, dataset, a1_path, gap_path, gap_info, output_path, feature_names,
              feature_source, config, metadata, model='gemini-2.5-flash', temperature=0,
              min_attributes=1, max_attributes=25, dry_run=False, concepts_path=None,
              bindings_path=None, prompt_out=None, raw_response_out=None):
    """Run two stages or replay saved responses; never read feature sample arrays."""
    vectors = load_json(str(a1_path))
    source_names = attribute_names(vectors)
    seen, unseen = gap_info['seen'], gap_info['unseen']
    if set(vectors) != set(seen + unseen):
        raise ValueError('A1 class vocabulary does not match this GAP split')
    if any(not isinstance(v, dict) or set(v) != set(source_names) or
           any(not isinstance(level, str) or level not in {'Low', 'Medium', 'High'} for level in v.values())
           for v in vectors.values()):
        raise ValueError('A1 vectors must share attributes with Low/Medium/High values')
    gap_hash = gap_info.get('a1_sha256')
    if isinstance(gap_hash, str) and gap_hash and gap_hash != _sha(a1_path):
        raise ValueError('GAP was calculated from different A1 content; recompute GAP')
    output = Path(output_path)
    stem = output.with_suffix('')
    concept_file = Path(str(stem) + '.concepts.json')
    concept_prompt = Path(prompt_out or str(stem) + '.prompt.txt')
    concept_raw = Path(raw_response_out or str(stem) + '.raw_response.txt')
    binding_prompt = Path(str(stem) + '.binding.prompt.txt')
    binding_raw = Path(str(stem) + '.binding.raw_response.txt')
    mapping_file = Path(str(stem) + '.feature_mapping.json')
    protected = {Path(p).resolve() for p in (a1_path, gap_path, concepts_path, bindings_path) if p}
    if Path(feature_source).is_file():
        protected.add(Path(feature_source).resolve())
    destinations = [output, concept_file, concept_prompt, concept_raw,
                    binding_prompt, binding_raw, mapping_file]
    resolved = [p.resolve() for p in destinations]
    # Reuse a saved stage response in place without overwriting that input.
    reusable = set()
    if concepts_path:
        reusable.update(p.resolve() for p in (concept_file, concept_raw)
                        if p.resolve() == Path(concepts_path).resolve())
    if bindings_path and binding_raw.resolve() == Path(bindings_path).resolve():
        reusable.add(binding_raw.resolve())
    immutable = {Path(a1_path).resolve(), Path(gap_path).resolve(), Path(feature_source).resolve()}
    if (len(set(resolved)) != len(resolved) or (protected & set(resolved)) - reusable or
            immutable & set(resolved)):
        raise ValueError('Output artifacts must be distinct and cannot overwrite inputs')
    context = dict(a1_names=source_names, seen=seen, unseen=unseen, dataset=dataset,
                   split_index=gap_info['split_index'], min_attributes=min_attributes,
                   max_attributes=max_attributes)
    prompt = build_prompt(dataset, seen, unseen, vectors, gap_info, metadata,
                          min_attributes=min_attributes, max_attributes=max_attributes,
                          feature_groups=group_context(dataset, feature_names, config, include_columns=False))
    write_text(prompt, str(concept_prompt))
    if dry_run and not concepts_path:
        return {'concept_prompt': str(concept_prompt), 'status': 'concept_prompt_only'}
    raw = (Path(concepts_path).read_text(encoding='utf-8') if concepts_path else
           call_gemini(prompt, model, temperature))
    if concept_raw.resolve() not in reusable:
        write_text(raw, str(concept_raw))
    concepts = extract_json_object(raw)
    validate_concepts(concepts, **context)
    if any(set(a) != {'name', 'definition', 'levels'} for a in concepts['attributes']):
        raise ValueError('Concept stage attributes must contain only name/definition/levels')
    if concept_file.resolve() not in reusable:
        dump_json(concepts, str(concept_file))
    prompt = build_binding_prompt(concepts, feature_names, config, metadata)
    write_text(prompt, str(binding_prompt))
    if dry_run and not bindings_path:
        return {'binding_prompt': str(binding_prompt), 'status': 'binding_prompt_only'}
    raw = (Path(bindings_path).read_text(encoding='utf-8') if bindings_path else
           call_gemini(prompt, model, temperature))
    if binding_raw.resolve() not in reusable:
        write_text(raw, str(binding_raw))
    plan = bind_features(concepts, extract_json_object(raw))
    mapping = validate_bound_plan(plan, feature_names=feature_names, groups_config=config, **context)
    meta = {'dataset': dataset, 'split_index': gap_info['split_index'], 'seen': seen,
            'unseen': unseen, 'a1_json': str(a1_path), 'a1_sha256': _sha(a1_path),
            'dknn_csv': str(gap_path), 'dknn_sha256': _sha(gap_path), 'gap': gap_info,
            'gap_a1_hash_verified': isinstance(gap_hash, str) and bool(gap_hash),
            'feature_source': feature_source,
            'feature_names_sha256': hashlib.sha256(json.dumps(list(feature_names)).encode()).hexdigest(),
            'groups_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'feature_mapping': str(mapping_file.resolve()), 'model': model,
            'concept_response_source': str(concepts_path) if concepts_path else 'live',
            'binding_response_source': str(bindings_path) if bindings_path else 'live',
            'validation': 'structural and feature availability; semantic review by LLM; no empirical gain claimed'}
    plan['_meta'] = meta
    mapping['_meta'] = meta
    dump_json(mapping, str(mapping_file))
    dump_json(plan, str(output))
    return {'plan': str(output), 'feature_mapping': str(mapping_file), 'status': 'validated'}


def main():
    parser = argparse.ArgumentParser(description='Split-specific conceptual refinement, then A2 group binding')
    def option(name, **kwargs):
        parser.add_argument('--' + name, '--' + name.replace('_', '-'), **kwargs)
    option('dataset_name', default='pamap2')
    option('a1_json', required=True)
    option('dknn_csv', required=True)
    option('output_json')
    option('output_root')
    option('all_splits', action='store_true')
    option('split_index', type=int)
    parser.add_argument('--seen', default='')
    parser.add_argument('--unseen', default='')
    option('metadata_json', default='')
    option('feature_names_npy')
    option('groups_json')
    option('templates_only', action='store_true', help='Dry-run prompts only; does not verify actual feature availability')
    option('concepts_json', help='Replay first-stage JSON response')
    option('bindings_json', help='Replay second-stage JSON response')
    option('responses_root', help='Batch replay: split_N/concepts.json and bindings.json')
    option('prompt_out')
    option('raw_response_out')
    option('min_attributes', type=int, default=1)
    option('max_attributes', type=int, default=DEFAULT_MAX_ATTRIBUTES_LIMIT)
    parser.add_argument('--model', default='gemini-2.5-flash')
    parser.add_argument('--temperature', type=float, default=0)
    option('dry_run', action='store_true')
    args = parser.parse_args()
    if args.bindings_json and not args.concepts_json:
        parser.error('--bindings-json requires --concepts-json to identify the frozen concepts')
    if not 1 <= args.min_attributes <= args.max_attributes <= DEFAULT_MAX_ATTRIBUTES_LIMIT:
        parser.error('Require 1 <= min_attributes <= max_attributes <= 25')
    if args.templates_only and (not args.dry_run or args.bindings_json or args.responses_root):
        parser.error('--templates-only is limited to prompt previews; no final binding validation')
    if args.all_splits:
        if not args.output_root or any((args.output_json, args.split_index is not None,
               args.seen, args.unseen, args.concepts_json, args.bindings_json, args.prompt_out, args.raw_response_out)):
            parser.error('--all-splits requires --output-root and no single-split options')
        indices = pd.read_csv(args.dknn_csv)['split_index'].tolist()
        if not indices or len(indices) != len(set(indices)):
            parser.error('Batch GAP must contain nonempty unique split indices')
    else:
        if not args.output_json or args.responses_root or args.output_root:
            parser.error('Single split requires --output-json; --responses-root is batch-only')
        indices = [args.split_index]
    config = load_groups(args.groups_json) if args.groups_json else load_groups()
    if args.templates_only:
        names = expand_feature_groups(args.dataset_name, config)['feature_names']
        feature_source = 'template_preview_only'
    else:
        path = Path(args.feature_names_npy) if args.feature_names_npy else load_dataset_spec(args.dataset_name).feature_names_path
        if not path.exists():
            raise FileNotFoundError(f'Actual feature-name file required: {path}. Use --feature-names-npy or --dry-run --templates-only for previews.')
        array = np.load(path, allow_pickle=True)
        if array.ndim != 1:
            raise ValueError('Feature-name array must be one-dimensional')
        names = list(map(str, array))
        feature_source = str(path.resolve())
    resolve_feature_groups(names, config, dataset=args.dataset_name)
    metadata = load_dataset_metadata(args.dataset_name, args.metadata_json)
    results = []
    for index in indices:
        gap = resolve_split_from_gap_csv(args.dknn_csv, parse_activity_list(args.seen),
                                         parse_activity_list(args.unseen), index)
        batch_dir = Path(args.output_root) / f"split_{gap['split_index']}" if args.all_splits else None
        responses = Path(args.responses_root) / f"split_{gap['split_index']}" if args.responses_root else None
        result = run_split(dataset=args.dataset_name, a1_path=args.a1_json, gap_path=args.dknn_csv,
            gap_info=gap, output_path=batch_dir / 'a2_attribute_plan.json' if batch_dir else args.output_json,
            feature_names=names, feature_source=feature_source, config=config, metadata=metadata,
            model=args.model, temperature=args.temperature, min_attributes=args.min_attributes,
            max_attributes=args.max_attributes, dry_run=args.dry_run,
            concepts_path=responses / 'concepts.json' if responses else args.concepts_json,
            bindings_path=responses / 'bindings.json' if responses else args.bindings_json,
            prompt_out=args.prompt_out, raw_response_out=args.raw_response_out)
        results.append({'split_index': gap['split_index'], **result})
        print(json.dumps(results[-1], ensure_ascii=False))
    if args.all_splits:
        dump_json(results, str(Path(args.output_root) / 'manifest.json'))


if __name__ == '__main__':
    main()
