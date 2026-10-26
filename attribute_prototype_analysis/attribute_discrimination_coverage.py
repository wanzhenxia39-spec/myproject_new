#!/usr/bin/env python3
"""Standalone Attribute Discrimination Coverage (Python standard library only).

R_k = count(i<j: p_ik != p_jk) / binom(C,2).
Only equality matters: any number of attributes or discrete levels is supported.
No distance, ordinal spacing, classifier output, or training is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
INPUT_ROOT = Path("/data/wanzhen/projects/myproject_new")
RESULTS_ROOT = Path("/home/wanzhen/attribute_prototype_analysis/results")
DATASETS = {"pamap2": "PAMAP2", "mhealth": "MHEALTH", "uschad": "USC-HAD",
            "wisdm": "WISDM", "dsads": "DSADS"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_vectors(path):
    """Read {activity: {attribute: level}} JSON or activity-first CSV."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        return json.loads(source.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object, parse_float=Decimal)
    if source.suffix.lower() != ".csv":
        raise ValueError("Input must be JSON or CSV")
    vectors = {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) < 2 or fields[0] != "activity" or len(set(fields)) != len(fields):
            raise ValueError("CSV needs unique headers: activity,<attribute>,...")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row length differs from header")
            activity = row.pop("activity")
            if activity in vectors:
                raise ValueError(f"Duplicate activity: {activity}")
            vectors[activity] = row
    return vectors


def _level(value, level_map):
    if isinstance(value, str) and value in level_map:
        value = level_map[value]
    if value is None or isinstance(value, (bool, list, dict)):
        raise ValueError(f"Invalid attribute level: {value!r}")
    if isinstance(value, str) and not value.strip():
        raise ValueError("Empty attribute level")
    try:
        numeric = Decimal(str(value))
    except InvalidOperation:
        if isinstance(value, str):
            return ("label", value)
        raise ValueError(f"Invalid attribute level: {value!r}") from None
    if not numeric.is_finite():
        raise ValueError("Non-finite attribute level")
    return ("number", numeric)


def compute_attribute_coverage(vectors: Mapping, *, level_map: Mapping | None = None) -> list[dict]:
    """Return per-attribute coverage and exact pair counts.

    Equal pairs = sum_l binom(n_l,2), so different pairs = binom(C,2)-equal pairs.
    This is exactly the requested i<j formula without constructing a C-by-C array.
    By default labels remain literal; numeric strings and numbers compare equally.
    Optional level_map explicitly harmonizes label encodings when required.
    """
    if not isinstance(vectors, Mapping) or len(vectors) < 2:
        raise ValueError("At least two activity prototypes are required")
    if any(not isinstance(name, str) or not name.strip() for name in vectors):
        raise ValueError("Activity names must be nonempty strings")
    first = next(iter(vectors.values()))
    if not isinstance(first, Mapping) or not first:
        raise ValueError("Each activity needs a nonempty attribute dictionary")
    attributes = list(first)
    if any(not isinstance(name, str) or not name.strip() for name in attributes):
        raise ValueError("Attribute names must be nonempty strings")
    if any(not isinstance(row, Mapping) or set(row) != set(attributes) for row in vectors.values()):
        raise ValueError("All activities must have exactly the same attributes")
    mapping = {} if level_map is None else level_map
    if not isinstance(mapping, Mapping):
        raise ValueError("level_map must be a dictionary")
    classes = len(vectors)
    pairs = classes * (classes - 1) // 2
    result = []
    for attribute in attributes:
        counts = Counter(_level(row[attribute], mapping) for row in vectors.values())
        equal = sum(count * (count - 1) // 2 for count in counts.values())
        different = pairs - equal
        result.append({"attribute": attribute, "different_pairs": different, "equal_pairs": equal,
                       "total_pairs": pairs, "coverage": different / pairs,
                       "single_attribute_collision_rate": equal / pairs})
    return result


def _assemble_matrix(metrics_by_dataset):
    if not metrics_by_dataset:
        raise ValueError("At least one dataset is required")
    datasets = list(metrics_by_dataset)
    attributes = [row["attribute"] for row in metrics_by_dataset[datasets[0]]]
    lookup = {}
    for dataset, rows in metrics_by_dataset.items():
        lookup[dataset] = {row["attribute"]: row["coverage"] for row in rows}
        if set(lookup[dataset]) != set(attributes):
            raise ValueError("Attributes differ across datasets: a shared matrix requires the same attribute names")
    return {"attributes": attributes, "datasets": datasets,
            "values": [[lookup[dataset][attribute] for dataset in datasets] for attribute in attributes]}


def build_coverage_matrix(vectors_by_dataset: Mapping, *, level_map: Mapping | None = None) -> dict:
    """Return {attributes, datasets, values}; rows=attributes, columns=datasets.

    Attribute order follows the first dataset; other datasets align by name.
    Different methods should be passed separately, since their attributes may differ.
    """
    if not isinstance(vectors_by_dataset, Mapping) or not vectors_by_dataset:
        raise ValueError("At least one dataset mapping is required")
    return _assemble_matrix({dataset: compute_attribute_coverage(vectors, level_map=level_map)
                             for dataset, vectors in vectors_by_dataset.items()})


def _output(path):
    output = Path(path).expanduser().resolve()
    if not output.is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError(f"Output must be under {RESULTS_ROOT}")
    parent = output
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path("/").stat().st_dev:
        raise ValueError("Analysis results must reside on the system disk")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid method/dataset identifier: {value!r}")
    return value


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.manifest:
        manifest = args.manifest.expanduser().resolve()
        entries = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        base = manifest.parent
    elif args.vectors:
        entries = [{"method": args.method, "dataset": args.dataset,
                    "vectors": str(args.vectors.expanduser().resolve())}]
        base = Path.cwd()
    else:
        entries = [{"method": "initial_a1_v1_5", "dataset": dataset,
                    "vectors": f"configs/attributes/activity_attribute_vectors_initial_{dataset}_v1_5.json"}
                   for dataset in args.datasets]
        base = args.input_root.expanduser().resolve()
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty JSON list")
    prepared, seen, class_sets = [], set(), {}
    by_method = defaultdict(dict)
    for entry in entries:
        method, dataset = _identifier(entry["method"]), _identifier(entry["dataset"])
        if (method, dataset) in seen:
            raise ValueError("Duplicate method/dataset entry")
        seen.add((method, dataset))
        source = (base / Path(entry["vectors"]).expanduser()).resolve()
        vectors = read_vectors(source)
        metrics = compute_attribute_coverage(vectors, level_map=entry.get("level_map"))
        if dataset in class_sets and class_sets[dataset] != set(vectors):
            raise ValueError(f"Methods must use identical activity sets for {dataset}")
        class_sets[dataset] = set(vectors)
        by_method[method][dataset] = metrics
        prepared.append((method, dataset, source, entry, vectors, metrics))
    # Reject incompatible matrices before writing any results.
    matrices = {method: _assemble_matrix(metrics) for method, metrics in by_method.items()}
    output = _output(args.output_dir)
    all_rows, summaries, provenance = [], [], []
    for method, dataset, source, entry, vectors, metrics in prepared:
        directory = _output(output / method / dataset)
        _csv(directory / "attribute_metrics.csv", metrics, list(metrics[0]))
        _json(directory / "summary.json", {
            "method": method, "dataset": dataset, "class_count": len(vectors), "attribute_count": len(metrics),
            "total_pairs": metrics[0]["total_pairs"],
            "mean_attribute_coverage": sum(row["coverage"] for row in metrics) / len(metrics),
        })
        summaries.append(json.loads((directory / "summary.json").read_text()))
        all_rows.extend({"method": method, "dataset": dataset, **row} for row in metrics)
        provenance.append({"method": method, "dataset": dataset, "path": str(source),
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "level_map": entry.get("level_map"), "activities": list(vectors),
                           "attributes": [row["attribute"] for row in metrics]})
    for method, matrix in matrices.items():
        directory = _output(output / method)
        # Display aliases only; identifiers in metadata and API remain unchanged.
        columns = [DATASETS.get(dataset, dataset) for dataset in matrix["datasets"]]
        if len(set(columns)) != len(columns) or "attribute" in columns:
            raise ValueError("Dataset display names are not unique or conflict with the attribute column")
        rows = [{"attribute": attribute, **dict(zip(columns, values))}
                for attribute, values in zip(matrix["attributes"], matrix["values"])]
        _csv(directory / "attribute_coverage_matrix.csv", rows, ["attribute", *columns])
        _csv(directory / "attribute_dataset_means.csv", [
            {"attribute": attribute, "dataset_count": len(columns),
             "mean_coverage_across_datasets": sum(values) / len(values)}
            for attribute, values in zip(matrix["attributes"], matrix["values"])],
            ["attribute", "dataset_count", "mean_coverage_across_datasets"])
    _csv(output / "all_attribute_metrics.csv", all_rows, list(all_rows[0]))
    _csv(output / "dataset_summary.csv", summaries, list(summaries[0]))
    _json(output / "run_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": provenance, "output_dir": str(output),
        "formula": "R_k=count(i<j:p_ik!=p_jk)/binom(C,2)",
        "equivalent_computation": "1-sum_l binom(n_kl,2)/binom(C,2)",
        "matrix_orientation": "rows=attributes, columns=datasets; align attributes by name",
        "level_policy": "exact equality only; no ordinal distance or automatic level normalization",
        "averaging": "attribute_dataset_means uses unweighted dataset means",
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    for method, matrix in matrices.items():
        print(f"{method}: {len(matrix['attributes'])} attributes x {len(matrix['datasets'])} datasets")
    print(f"Outputs: {output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--vectors", type=Path, help="One custom JSON/CSV prototype file")
    source.add_argument("--manifest", type=Path, help="Batch list of method/dataset/vector entries")
    parser.add_argument("--method", default="custom", help="Method identifier for --vectors")
    parser.add_argument("--dataset", default="custom", help="Dataset identifier for --vectors")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "attribute_coverage" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
