"""Compute pre-refinement GAP from current A1 vectors and export planner inputs.

The default distance is equal-weight normalized ordinal prototype distance.
This module validates A1 inputs, selects splits and serializes planner diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from zshar.datasets import available_datasets, load_dataset_spec
from zshar.paths import DEFAULT_RUNTIME_ROOT
from zshar.semantics.completion import validate_llm_output
from zshar.semantics.gap import (
    DEFAULT_GAP_DISTANCE, GAP_DISTANCE_METRICS,
    assign_difficulty, attribute_distance, compute_gap, vectorize_prototypes,
)
from zshar.semantics.ontology import initial_attribute_names
from zshar.splits import default_manifest_paths, generate_activity_splits, load_split_manifest


def planner_gap_row(split, activity_vectors, *, k=2, metric=DEFAULT_GAP_DISTANCE) -> dict:
    """Add the CSV fields consumed by planner.resolve_split_from_gap_csv()."""
    if k <= 0:
        raise ValueError("top-k must be positive")
    seen, unseen = list(split.seen_activities), list(split.unseen_activities)
    if not seen or not unseen or set(seen) & set(unseen):
        raise ValueError("A GAP split needs nonempty, disjoint seen and unseen activities")
    if len(set(seen)) != len(seen) or len(set(unseen)) != len(unseen):
        raise ValueError("Duplicate activities in GAP split")
    if set(seen) | set(unseen) != set(activity_vectors):
        raise ValueError("Seen/unseen activities must cover the A1 activity vocabulary exactly")
    row = compute_gap(split, activity_vectors, k=k, metric=metric)
    attributes, numeric_vectors = vectorize_prototypes(activity_vectors)
    weights = [1.0] * len(attributes)
    nearest = {}
    for activity in unseen:
        distances = [(other, attribute_distance(numeric_vectors[activity], numeric_vectors[other],
                                                weights, metric=metric)) for other in seen]
        distances.sort(key=lambda item: item[1])
        nearest[activity] = distances[:min(k, len(seen))]
    return {
        **row,
        "seen_classes": "|".join(seen),
        "unseen_classes": "|".join(unseen),
        "seen_key": "|".join(sorted(seen)),
        "unseen_key": "|".join(sorted(unseen)),
        "top_k": k,
        "effective_k": min(k, len(seen)),
        "distance_metric": metric,
        "D_kNN_per_unseen_json": json.dumps(row["D_kNN_per_unseen"], ensure_ascii=False),
        "nearest_seen_topk_json": json.dumps(nearest, ensure_ascii=False),
        "U_NN_per_unseen_json": json.dumps(row["U_NN_per_unseen"], ensure_ascii=False),
    }


def run_gap_analysis(*, dataset, dataset_config=None, a1_json=None, split_manifest=None,
                     seed=42, max_splits=None, top_k=2, distance_metric=DEFAULT_GAP_DISTANCE, output=None) -> Path:
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    if max_splits is not None and max_splits <= 0:
        raise ValueError("max-splits must be positive")
    if distance_metric not in GAP_DISTANCE_METRICS:
        raise ValueError(f"distance-metric must be one of {GAP_DISTANCE_METRICS}")
    spec = load_dataset_spec(dataset, dataset_config)
    vector_path = Path(a1_json).expanduser().resolve() if a1_json else spec.attributes_path.resolve()
    if not vector_path.is_file():
        raise FileNotFoundError(
            f"A1 activity vectors not found: {vector_path}. Generate them with "
            f"python -m zshar.semantics.initial_vectors --dataset {spec.key}, "
            "or specify --a1-json. initial_attributes.json contains definitions, not activity vectors."
        )
    # Read on every run: no cached vectors or fallback to the old ten-attribute files.
    vector_bytes = vector_path.read_bytes()
    attribute_names = initial_attribute_names()
    vectors = validate_llm_output(json.loads(vector_bytes), spec.activity_names, attribute_names)
    default_split, _ = default_manifest_paths(spec, scenario="zsl", seed=seed)
    manifest = Path(split_manifest).expanduser().resolve() if split_manifest else default_split
    if split_manifest and not manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest}")
    if manifest.is_file():
        splits = load_split_manifest(manifest, spec)
    else:
        splits = generate_activity_splits(spec, seed=seed, max_splits=max_splits)
    if max_splits is not None:
        splits = splits[:max_splits]
    if not splits:
        raise ValueError("No splits selected for GAP analysis")
    indices = [split.index for split in splits]
    if len(indices) != len(set(indices)):
        raise ValueError("Split indices must be unique for planner CSV lookup")
    rows = [planner_gap_row(split, vectors, k=top_k, metric=distance_metric) for split in splits]
    # U_NN is undefined for a single unseen class. D_kNN remains valid.
    if all(pd.notna(row["JointGap_DkNN_UConf"]) for row in rows):
        frame = assign_difficulty(rows)
    else:
        frame = pd.DataFrame(rows)
        frame["difficulty"] = "Unranked"
    vector_hash = hashlib.sha256(vector_bytes).hexdigest()
    frame["a1_json"] = str(vector_path)
    frame["a1_sha256"] = vector_hash
    # Serialize structured original columns as valid JSON, not Python repr.
    for column in ("seen_activities", "unseen_activities", "D_kNN_per_unseen", "U_NN_per_unseen"):
        frame[column] = frame[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    destination = Path(output).expanduser().resolve() if output else (
        DEFAULT_RUNTIME_ROOT / "gap" / spec.key / f"a1_gap_seed{seed}.csv"
    )
    metadata_path = destination.with_suffix(".meta.json")
    protected_inputs = {vector_path, manifest.resolve()}
    if destination.resolve() in protected_inputs or metadata_path.resolve() in protected_inputs:
        raise ValueError("GAP outputs must not overwrite an input vector or split manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    metadata = {
        "dataset": spec.key, "attribute_set": "A1", "attributes": attribute_names,
        "a1_json": str(vector_path), "a1_sha256": vector_hash,
        "activity_vectors": vectors,
        "split_manifest": str(manifest.resolve()) if manifest.is_file() else None,
        "split_source": "manifest" if manifest.is_file() else "generated_activity_splits",
        "splits": [{"split_index": s.index, "seen_activities": list(s.seen_activities),
                    "unseen_activities": list(s.unseen_activities)} for s in splits],
        "seed": seed, "top_k": top_k, "distance_metric": distance_metric,
        "attribute_weighting": "uniform",
        "distance_formula": (
            "sqrt((1/K_A) * sum_k(((a_i,k - a_j,k) / 2)^2))"
            if distance_metric == "euclidean" else
            "(1/K_A) * sum_k(abs(a_i,k - a_j,k) / 2)"
        ),
        "formula_ordinal_values": {"Low": 0, "Medium": 1, "High": 2},
        "ordinal_values": {"Low": 0.0, "Medium": 0.5, "High": 1.0},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{spec.key}: wrote {len(rows)} A1 GAP rows to {destination}")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-config")
    parser.add_argument("--a1-json", "--a1_json", dest="a1_json")
    parser.add_argument("--split-manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--distance-metric", choices=GAP_DISTANCE_METRICS, default=DEFAULT_GAP_DISTANCE)
    parser.add_argument("--output", help="Defaults to runtime/gap/<dataset>/a1_gap_seed<seed>.csv")
    args = parser.parse_args(argv)
    run_gap_analysis(**vars(args))


if __name__ == "__main__":
    main()
