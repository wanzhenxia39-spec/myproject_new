#!/usr/bin/env python3
"""Reproducible paper figures/metrics from class-level attribute prototypes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = Path("/data/wanzhen/projects/myproject_new")
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd

from exact_apcr import result_output

from discriminability import (
    DEFAULT_LEVEL_MAP, analyze_prototypes, load_prototypes, plot_blue_heatmap,
)

DATASET_NAMES = {"pamap2": "PAMAP2", "mhealth": "MHEALTH", "uschad": "USC-HAD",
                 "wisdm": "WISDM", "dsads": "DSADS"}


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Use letters, digits, underscore or hyphen for identifiers: {value!r}")
    return value



def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_entries(args):
    if args.manifest:
        manifest = args.manifest.expanduser().resolve()
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        base = manifest.parent
    else:
        entries = [{"method": "initial_a1_v1_5", "dataset": dataset,
                    "vectors": f"configs/attributes/activity_attribute_vectors_initial_{dataset}_v1_5.json"}
                   for dataset in args.datasets]
        base = args.input_root.expanduser().resolve()
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty JSON list")
    seen = set()
    prepared = []
    class_sets = {}
    for item in entries:
        method, dataset = safe_id(item["method"]), safe_id(item["dataset"])
        if (method, dataset) in seen:
            raise ValueError("One prototype table per method/dataset is allowed; do not mix splits")
        seen.add((method, dataset))
        source = (base / item["vectors"]).expanduser().resolve()
        level_map = item.get("level_map", DEFAULT_LEVEL_MAP)
        value_range = tuple(item.get("value_range", [0, 2]))
        table = load_prototypes(source, level_map=level_map, value_range=value_range)
        if dataset in class_sets and class_sets[dataset] != set(table.activities):
            raise ValueError(f"Methods must use identical activity sets for {dataset}")
        class_sets[dataset] = set(table.activities)
        if not args.manifest and len(table.attributes) != 9:
            raise ValueError(f"Expected latest nine-dimensional A1 for {dataset}")
        prepared.append((method, dataset, source, level_map, value_range, table))
    return prepared


def run(args):
    # Validate every input before creating any outputs.
    entries = read_entries(args)
    selected = set(args.distance_datasets)
    available = {entry[1] for entry in entries}
    if not selected <= available:
        raise ValueError(f"No inputs for distance plots: {sorted(selected - available)}")
    output = result_output(args.output_dir)
    # All plotting configuration/cache writes stay in the system-disk results directory.
    cache = result_output(output / "_cache")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["TMPDIR"] = str(cache)
    records, all_attributes, provenance = [], [], []
    coverage_by_method = {}
    for method, dataset, source, level_map, value_range, table in entries:
        directory = result_output(output / method / dataset)
        result = analyze_prototypes(table)
        result["distance_matrix"].to_csv(directory / "distance_matrix.csv", index_label="activity")
        attributes = result["attribute_metrics"]
        attributes.to_csv(directory / "attribute_metrics.csv", index=False)
        pairs = result["pair_metrics"]
        pairs.to_csv(directory / "activity_pairs.csv", index=False)
        pairs.loc[pairs.exact_collision == 1].to_csv(directory / "exact_collision_pairs.csv", index=False)
        pd.DataFrame(table.values, index=table.activities, columns=table.attributes).to_csv(
            directory / "normalized_prototypes.csv", index_label="activity")
        record = {"method": method, "dataset": dataset, **result["summary"]}
        write_json(directory / "summary.json", record)
        records.append(record)
        all_attributes.append(attributes.assign(method=method, dataset=dataset))
        coverage_by_method.setdefault(method, {})[dataset] = attributes.set_index("attribute")["coverage"]
        provenance.append({"method": method, "dataset": dataset, "path": str(source),
                           "sha256": sha256(source), "level_map": level_map,
                           "value_range": value_range, "activities": table.activities,
                           "attributes": table.attributes})
        if dataset in selected:
            plot_blue_heatmap(result["distance_matrix"], directory / "prototype_distance_heatmap",
                              colorbar_label="Normalized ordinal L1 distance",
                              title=f"{DATASET_NAMES.get(dataset, dataset)} ({method})" if args.titles else None,
                              formats=tuple(args.formats), dpi=args.dpi)

    summary = pd.DataFrame(records)
    summary.to_csv(output / "dataset_summary.csv", index=False)
    pd.concat(all_attributes, ignore_index=True).to_csv(output / "all_attribute_metrics.csv", index=False)
    aggregate = []
    for method, group in summary.groupby("method", sort=False):
        aggregate.append({
            "method": method, "dataset_count": len(group),
            "macro_mean_exact_apcr": float(group.exact_apcr.mean()),
            "pair_weighted_exact_apcr": float(group.exact_collision_pairs.sum() / group.total_pairs.sum()),
            "macro_mean_attribute_coverage": float(group.mean_attribute_coverage.mean()),
            "macro_mean_single_attribute_collision_rate": float(group.mean_single_attribute_collision_rate.mean()),
        })
    pd.DataFrame(aggregate).to_csv(output / "method_summary.csv", index=False)
    for method, columns in coverage_by_method.items():
        method_dir = result_output(output / method)
        # Different methods can have different K. Never invent an alignment.
        reference = next(iter(columns.values())).index
        if not all(set(column.index) == set(reference) for column in columns.values()):
            write_json(method_dir / "coverage_matrix_skipped.json", {
                "reason": "Attributes differ across datasets; see per-dataset attribute_metrics.csv"})
            continue
        matrix = pd.DataFrame({DATASET_NAMES.get(ds, ds): column.reindex(reference)
                               for ds, column in columns.items()})
        matrix.index.name = "attribute"
        matrix.to_csv(method_dir / "attribute_coverage_matrix.csv")
        (1 - matrix).to_csv(method_dir / "single_attribute_collision_matrix.csv")
        pd.DataFrame({
            "dataset_count": len(matrix.columns),
            "mean_coverage_across_datasets": matrix.mean(axis=1),
            "mean_single_attribute_collision_rate_across_datasets": (1 - matrix).mean(axis=1),
        }).to_csv(method_dir / "attribute_dataset_means.csv")
        plot_blue_heatmap(matrix, method_dir / "attribute_coverage_heatmap",
                          colorbar_label="Attribute discrimination coverage",
                          title=method if args.titles else None, coverage=True,
                          formats=tuple(args.formats), dpi=args.dpi)
    import matplotlib
    write_json(output / "run_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(output),
        "inputs": provenance, "class_scope": "all activities in each source, no split",
        "distance": "mean_k(abs(p_ik-p_jk)/(declared_max-declared_min))",
        "coverage": "count(i<j: p_ik != p_jk) / binom(C,2)",
        "single_attribute_collision": "1 - coverage; NOT full-vector exact APCR",
        "exact_apcr": "count(i<j: all_k(p_ik == p_jk)) / binom(C,2)",
        "default_average": "unweighted macro mean across datasets",
        "colormap": "Blues", "color_limits": [0, 1], "distance_plot_datasets": args.distance_datasets,
        "formats": args.formats, "dpi": args.dpi,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "code_sha256": {str(path): sha256(path) for path in (
            Path(__file__), CODE_ROOT / "discriminability.py", CODE_ROOT / "exact_apcr.py",
            CODE_ROOT / "prototype_distance_heatmap.py", CODE_ROOT / "attribute_discrimination_coverage.py")},
    })
    print(summary[["dataset", "method", "exact_collision_pairs", "total_pairs", "exact_apcr",
                   "mean_single_attribute_collision_rate"]].to_string(index=False))
    print(pd.DataFrame(aggregate).to_string(index=False))
    print(f"Outputs: {output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_NAMES), default=list(DATASET_NAMES),
                        help="Default: latest nine-attribute A1 of all five datasets")
    parser.add_argument("--manifest", type=Path,
                        help="Custom method inputs; paths are relative to the manifest")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                        help="Input project root; independent of the code location")
    parser.add_argument("--distance-datasets", nargs="*", default=["pamap2"],
                        help="Only these datasets receive distance plots (all distance CSVs are saved)")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/wanzhen/attribute_prototype_analysis/results") /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--titles", action="store_true", help="Add titles; off by default for paper captions")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    run(arguments)
