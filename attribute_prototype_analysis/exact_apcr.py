#!/usr/bin/env python3
"""Standalone exact prototype collision rate (standard library only).

APCR = count(i<j: every coordinate of prototype i equals prototype j) / C(C-1)/2.
No distance, tolerance, plotting, training, or single-attribute averaging is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
DEFAULT_INPUT_ROOT = Path("/data/wanzhen/projects/myproject_new")
DATASETS = ("pamap2", "mhealth", "uschad", "wisdm", "dsads")
DEFAULT_LEVEL_MAP = {"Low": 0, "Medium": 1, "High": 2}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_vectors(path):
    """Read activity -> attribute -> level JSON, or activity-first CSV."""
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


def _canonical_level(value, level_map):
    if isinstance(value, str) and value in level_map:
        value = level_map[value]
    if value is None or isinstance(value, (bool, list, dict)):
        raise ValueError(f"Invalid attribute level: {value!r}")
    if isinstance(value, str) and not value.strip():
        raise ValueError("Empty attribute level")
    # Decimal preserves exact numeric equality without rounding/tolerances.
    try:
        numeric = Decimal(str(value))
    except InvalidOperation:
        if isinstance(value, str):
            return ("label", value)
        raise ValueError(f"Invalid attribute level: {value!r}") from None
    if not numeric.is_finite():
        raise ValueError("Non-finite attribute level")
    return ("number", numeric)


def compute_exact_apcr(vectors: Mapping, *, level_map: Mapping | None = None) -> dict:
    """Return full-vector exact APCR, counts and the actual colliding class pairs.

    Inputs have shape {activity: {attribute: level}}. Low/Medium/High defaults to
    0/1/2; other categorical labels are compared exactly. Numeric normalization
    is unnecessary for exact equality. Pass level_map={} to keep all labels literal.
    """
    if not isinstance(vectors, Mapping) or len(vectors) < 2:
        raise ValueError("At least two activity prototypes are required")
    activities = list(vectors)
    if any(not isinstance(name, str) or not name.strip() for name in activities):
        raise ValueError("Activity names must be nonempty strings")
    first = vectors[activities[0]]
    if not isinstance(first, Mapping) or not first:
        raise ValueError("Each activity needs a nonempty attribute dictionary")
    attributes = list(first)
    if any(not isinstance(name, str) or not name.strip() for name in attributes):
        raise ValueError("Attribute names must be nonempty strings")
    if any(not isinstance(row, Mapping) or set(row) != set(attributes) for row in vectors.values()):
        raise ValueError("All activities must have exactly the same attributes")
    mapping = DEFAULT_LEVEL_MAP if level_map is None else level_map
    if not isinstance(mapping, Mapping):
        raise ValueError("level_map must be a dictionary")
    groups = defaultdict(list)
    for activity, row in vectors.items():
        key = tuple(_canonical_level(row[attribute], mapping) for attribute in attributes)
        groups[key].append(activity)
    collision_pairs = [dict(activity_i=left, activity_j=right)
                       for group in groups.values() for left, right in combinations(group, 2)]
    total_pairs = len(activities) * (len(activities) - 1) // 2
    return {
        "class_count": len(activities), "attribute_count": len(attributes),
        "total_pairs": total_pairs, "exact_collision_pairs": len(collision_pairs),
        "exact_apcr": len(collision_pairs) / total_pairs,
        "collision_pairs": collision_pairs,
    }


def result_output(path):
    """Keep this explicitly authorized analysis's outputs on the system disk."""
    output = Path(path).expanduser().resolve()
    root = Path('/home/wanzhen/attribute_prototype_analysis/results').resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must be under /home/wanzhen/attribute_prototype_analysis/results')
    parent = output
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path('/').stat().st_dev:
        raise ValueError('Analysis outputs must reside on the system disk')
    output.mkdir(parents=True, exist_ok=True)
    return output


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid dataset/method identifier: {value!r}")
    return value


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(path, rows, fields):
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
        entries = [{"method": args.method, "dataset": args.dataset, "vectors": str(args.vectors.resolve())}]
        base = Path.cwd()
    else:
        entries = [{"method": "initial_a1_v1_5", "dataset": ds,
                    "vectors": f"configs/attributes/activity_attribute_vectors_initial_{ds}_v1_5.json"}
                   for ds in args.datasets]
        base = args.input_root.expanduser().resolve()
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty JSON list")
    prepared, seen, class_sets = [], set(), {}
    for entry in entries:
        method, dataset = _identifier(entry["method"]), _identifier(entry["dataset"])
        if (method, dataset) in seen:
            raise ValueError("Duplicate method/dataset in manifest")
        seen.add((method, dataset))
        source = (base / Path(entry["vectors"]).expanduser()).resolve()
        vectors = read_vectors(source)
        result = compute_exact_apcr(vectors, level_map=entry.get("level_map"))
        if dataset in class_sets and class_sets[dataset] != set(vectors):
            raise ValueError(f"Methods must use identical activity sets for {dataset}")
        class_sets[dataset] = set(vectors)
        prepared.append((method, dataset, source, entry, vectors, result))
    output = result_output(args.output_dir)
    rows, provenance = [], []
    for method, dataset, source, entry, vectors, result in prepared:
        directory = result_output(output / method / dataset)
        pairs = result.pop("collision_pairs")
        row = {"method": method, "dataset": dataset, **result}
        rows.append(row)
        _write_json(directory / "summary.json", row)
        _write_csv(directory / "exact_collision_pairs.csv", pairs, ["activity_i", "activity_j"])
        provenance.append({"method": method, "dataset": dataset, "path": str(source),
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "level_map": entry.get("level_map", DEFAULT_LEVEL_MAP),
                           "activities": list(vectors), "attributes": list(next(iter(vectors.values())))})
    _write_csv(output / "dataset_summary.csv", rows, list(rows[0]))
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    averages = [{
        "method": method, "dataset_count": len(group),
        "macro_mean_exact_apcr": sum(row["exact_apcr"] for row in group) / len(group),
        "pair_weighted_exact_apcr": sum(row["exact_collision_pairs"] for row in group) /
                                    sum(row["total_pairs"] for row in group),
    } for method, group in by_method.items()]
    _write_csv(output / "method_summary.csv", averages, list(averages[0]))
    _write_json(output / "run_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": provenance,
        "output_dir": str(output), "formula": "count(i<j: all_k(p_ik == p_jk)) / binom(C,2)",
        "default_average": "unweighted macro mean across datasets", "tolerance": None,
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version.split()[0],
    })
    for row in rows:
        print(f"{row['method']}/{row['dataset']}: {row['exact_collision_pairs']}/{row['total_pairs']} = {row['exact_apcr']:.8f}")
    for average in averages:
        print(f"{average['method']}: macro mean APCR = {average['macro_mean_exact_apcr']:.8f}")
    print(f"Outputs: {output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--vectors", type=Path, help="One custom prototype JSON/CSV")
    source.add_argument("--manifest", type=Path, help="Custom method/dataset list; paths relative to manifest")
    parser.add_argument("--dataset", default="custom", help="Dataset identifier for --vectors")
    parser.add_argument("--method", default="custom", help="Method identifier for --vectors")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/wanzhen/attribute_prototype_analysis/results/exact_apcr") /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
