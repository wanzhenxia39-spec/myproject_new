"""Summarize unseen-test attribute accuracy across split-specific A2 schemas."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence


def _write_csv(path, rows, columns):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_attribute_accuracy(
    run_dir: str | Path, *, expected_split_indices: Sequence[int],
    vector_inputs: Mapping[int, Mapping] | None = None, require_complete=False,
    expected_sample_count: int | None = None,
) -> dict:
    """Average each attribute over splits where it exists, with explicit coverage.

    Overall accuracy is mean_s(mean_a(accuracy[s,a])), so larger A2 schemas do
    not give their split more weight. Absent attributes are never scored as zero.
    Partial exports are marked incomplete and can be updated after a resumed run.
    """
    root = Path(run_dir)
    expected = list(map(int, expected_split_indices))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected split indices must be nonempty and unique")
    long_rows, split_rows, by_attribute, definitions = [], [], {}, {}
    for index in expected:
        path = root / f"split_{index:03d}" / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        per_attribute = metrics.get("per_attribute")
        count = int(metrics.get("sample_count", 0))
        if not isinstance(per_attribute, dict) or not per_attribute or count <= 0:
            raise ValueError(f"Invalid per-attribute metrics or sample count: {path}")
        if expected_sample_count is not None and count != expected_sample_count:
            raise ValueError(f"Unexpected test sample count in {path}: {count}")
        if vector_inputs is not None:
            source = vector_inputs[index]
            provenance = metrics.get("attribute_vectors", {})
            if provenance.get("sha256") != source["sha256"]:
                raise ValueError(f"Attribute metrics came from different vectors: {path}")
            if set(per_attribute) != set(source["attributes"]):
                raise ValueError(f"Metric attributes disagree with the split schema: {path}")
            for name, definition in source.get("semantic_definitions", {}).items():
                fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
                definitions.setdefault(name, set()).add(fingerprint)
        accuracies = []
        for name, values in per_attribute.items():
            accuracy = float(values["accuracy"])
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError(f"Invalid accuracy for {name}: {path}")
            row = {"split_index": index, "attribute": name, "accuracy": accuracy, "sample_count": count}
            long_rows.append(row)
            by_attribute.setdefault(name, []).append(row)
            accuracies.append(accuracy)
        split_rows.append({"split_index": index, "attribute_count": len(accuracies),
                           "sample_count": count, "attribute_mean_accuracy": statistics.mean(accuracies)})
    completed = {row["split_index"] for row in split_rows}
    missing = [index for index in expected if index not in completed]
    if require_complete and missing:
        raise ValueError(f"Missing test metrics for splits: {missing}")
    attribute_rows = []
    for name, rows in sorted(by_attribute.items()):
        accuracies = [row["accuracy"] for row in rows]
        present = {row["split_index"] for row in rows}
        attribute_rows.append({
            "attribute": name, "mean_accuracy": statistics.mean(accuracies),
            "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "min_accuracy": min(accuracies), "max_accuracy": max(accuracies),
            "present_split_count": len(rows), "expected_split_count": len(expected),
            "coverage": len(rows) / len(expected),
            "present_split_indices": json.dumps([i for i in expected if i in present]),
            "absent_split_indices": json.dumps([i for i in expected if i in completed and i not in present]),
            "unevaluated_split_indices": json.dumps(missing),
            "test_sample_count": sum(row["sample_count"] for row in rows),
            "semantic_definition_variants": len(definitions[name]) if name in definitions else None,
        })
    means = [row["attribute_mean_accuracy"] for row in split_rows]
    summary = {
        "expected_split_count": len(expected), "completed_split_count": len(completed),
        "complete": not missing, "missing_split_indices": missing,
        "attribute_mean_accuracy": statistics.mean(means) if means else None,
        "attribute_mean_accuracy_std": statistics.stdev(means) if len(means) > 1 else (0.0 if means else None),
        "overall_aggregation": "mean over splits of the mean test accuracy over that split's attributes",
        "per_attribute_aggregation": "mean over evaluated splits containing the attribute; absent attributes excluded",
        "accuracy_scale": "fraction in [0,1]", "standard_deviation": "sample (ddof=1)",
        "per_attribute": attribute_rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "attribute_accuracy_by_split.csv", long_rows,
               ["split_index", "attribute", "accuracy", "sample_count"])
    _write_csv(root / "split_attribute_accuracy.csv", split_rows,
               ["split_index", "attribute_count", "sample_count", "attribute_mean_accuracy"])
    _write_csv(root / "per_attribute_accuracy_summary.csv", attribute_rows, [
        "attribute", "mean_accuracy", "std_accuracy", "min_accuracy", "max_accuracy",
        "present_split_count", "expected_split_count", "coverage", "present_split_indices",
        "absent_split_indices", "unevaluated_split_indices", "test_sample_count", "semantic_definition_variants"])
    temporary = root / "attribute_accuracy_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(root / "attribute_accuracy_summary.json")
    return summary
