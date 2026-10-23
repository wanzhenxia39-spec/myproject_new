"""Recompute A1/A2 semantic GAP from saved prototypes without invoking an LLM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from zshar.datasets import available_datasets, load_dataset_spec
from zshar.semantics.completion import get_attribute_names, validate_llm_output
from zshar.semantics.gap import DEFAULT_GAP_DISTANCE, assign_difficulty
from zshar.semantics.gap_analysis import planner_gap_row, run_gap_analysis
from zshar.splits import default_manifest_paths, load_split_manifest


GAP_METRICS = ("D_kNN", "U_NN", "U_Conf", "JointGap_DkNN_UConf")


def compare_refinement_gap(
    *, dataset, a2_vectors_root, output_dir, a1_json=None,
    split_manifest=None, seed=42, top_k=2,
) -> Path:
    """Export per-split diagnostics, differences and provenance for both sets.

    Each split's A2 schema is validated independently, so differing attribute
    counts are allowed. Saved vector files and experiment results are read-only.
    """
    spec = load_dataset_spec(dataset)
    a1_path = Path(a1_json).expanduser().resolve() if a1_json else spec.attributes_path.resolve()
    manifest = (Path(split_manifest).expanduser().resolve() if split_manifest else
                default_manifest_paths(spec, scenario="zsl", seed=seed)[0])
    root, destination = Path(a2_vectors_root).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    splits = load_split_manifest(manifest, spec)
    a1 = json.loads(a1_path.read_text(encoding="utf-8"))
    # Validate all A2 inputs before writing any reports.
    pairs, a2_rows, provenance = [], [], []
    max_mad_difference = 0.0
    for split in splits:
        vector_path = root / f"split_{split.index}" / "activity_attribute_vectors_a2.json"
        raw = vector_path.read_bytes()
        payload = json.loads(raw)
        plan_path = vector_path.parent / "a2_attribute_plan.json"
        attributes = (get_attribute_names(json.loads(plan_path.read_text(encoding="utf-8")))
                      if plan_path.is_file() else list(next(iter(payload.values()))))
        a2 = validate_llm_output(payload, list(spec.activity_names), attributes)
        before = planner_gap_row(split, a1, k=top_k)
        after = planner_gap_row(split, a2, k=top_k)
        after.update({"attribute_count": len(attributes), "attribute_vectors_path": str(vector_path),
                      "attribute_vectors_sha256": hashlib.sha256(raw).hexdigest()})
        a2_rows.append(after)
        row = {"dataset": spec.key, "split_index": split.index,
               "distance_metric": DEFAULT_GAP_DISTANCE,
               "A1_attribute_count": len(next(iter(a1.values()))), "A2_attribute_count": len(attributes)}
        for metric in GAP_METRICS:
            row[f"A1_{metric}"] = before[metric]
            row[f"A2_{metric}"] = after[metric]
            row[f"delta_{metric}"] = after[metric] - before[metric]
        pairs.append(row)
        for vectors, current in ((a1, before), (a2, after)):
            old = planner_gap_row(split, vectors, k=top_k, metric="mad")
            for metric in GAP_METRICS:
                max_mad_difference = max(max_mad_difference, abs(old[metric] - current[metric]))
        provenance.append({"split_index": split.index, "path": str(vector_path),
                           "sha256": after["attribute_vectors_sha256"], "attributes": attributes})

    output_files = [destination / name for name in (
        "a1_gap.csv", "a1_gap.meta.json", "a2_gap.csv", "comparison.csv", "summary.json")]
    inputs = {a1_path, manifest.resolve()}
    inputs.update(Path(item["path"]) for item in provenance)
    inputs.update(root.glob("split_*/a2_attribute_plan.json"))
    if inputs.intersection(output_files):
        raise ValueError("GAP comparison outputs must not overwrite input artifacts")
    run_gap_analysis(dataset=spec.key, a1_json=a1_path, split_manifest=manifest,
                     seed=seed, top_k=top_k, output=destination / "a1_gap.csv")
    if all(pd.notna(row["JointGap_DkNN_UConf"]) for row in a2_rows):
        a2_frame = assign_difficulty(a2_rows)
    else:
        a2_frame = pd.DataFrame(a2_rows)
        a2_frame["difficulty"] = "Unranked"
    for column in ("seen_activities", "unseen_activities", "D_kNN_per_unseen", "U_NN_per_unseen"):
        a2_frame[column] = a2_frame[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    a2_frame.to_csv(destination / "a2_gap.csv", index=False)
    comparison = pd.DataFrame(pairs)
    comparison.to_csv(destination / "comparison.csv", index=False)
    summary = {
        "dataset": spec.key, "seed": seed, "split_count": len(splits), "top_k": top_k,
        "distance_metric": DEFAULT_GAP_DISTANCE, "attribute_weighting": "uniform",
        "distance_formula": "(1/K_A) * sum_k(abs(a_i,k - a_j,k) / 2)",
        "formula_ordinal_values": {"Low": 0, "Medium": 1, "High": 2},
        "a1_json": str(a1_path), "a1_sha256": hashlib.sha256(a1_path.read_bytes()).hexdigest(),
        "split_manifest": str(manifest),
        "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "a2_inputs": provenance,
        "vector_source": "existing_saved_vectors; no LLM generation or classifier training",
        "maximum_absolute_difference_from_uniform_mad": max_mad_difference,
        "metrics": {
            metric: {prefix: float(comparison[f"{prefix}_{metric}"].mean())
                     for prefix in ("A1", "A2", "delta")}
            for metric in GAP_METRICS
        },
    }
    # Strict JSON: single-unseen GAP metrics are undefined, represented as null.
    for values in summary["metrics"].values():
        for key, value in values.items():
            if pd.isna(value):
                values[key] = None
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{spec.key}: wrote {len(pairs)} A1/A2 comparisons to {destination}")
    return destination / "comparison.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--a1-json")
    parser.add_argument("--a2-vectors-root", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    compare_refinement_gap(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
