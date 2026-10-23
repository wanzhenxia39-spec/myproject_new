#!/usr/bin/env python3
"""Run the PAMAP2 initial-A1 binary/ordinal-5 GZSL tests on GPU 0.

The two variants keep the A1 v1.5 nine-attribute schema and differ only in
the supplied attribute-vector level setting.  They use the seed-42 manifests
in ``split/gzsl_20``,
independent AutoGluon default ensembles, ordinal-probability prototype
matching, and the matched temperature + seen-penalty GZSL calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE_SCRIPT = PROJECT / "scripts/run_pamap2_a2_gzsl_autogluon_default_ensemble.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "pamap2_gzsl_default_ensemble_level_base", BASE_SCRIPT
)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

LEVEL_LABELS = {
    2: ("Low", "High"),
    5: ("Very Low", "Low", "Medium", "High", "Very High"),
}
VECTOR_PATHS = {
    2: PROJECT
    / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5_binary.json",
    5: PROJECT
    / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5_ordinal_5.json",
}
RUN_NAMES = {
    level_count: (
        "pamap2_gzsl_autogluon_default_ensemble_initial_a1_v1_5_"
        f"{level_count}level_independent_calibrated_grid_cal3_t300_"
        "random20_eval20_seen20_seed42_gpu0"
    )
    for level_count in LEVEL_LABELS
}
RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def dataset_spec(level_count: int) -> Any:
    from zshar.datasets import load_dataset_spec

    original = load_dataset_spec(base.DATASET)
    vector_path = VECTOR_PATHS[level_count].resolve()
    raw = dict(original.raw)
    raw["attributes"] = str(vector_path)
    return replace(original, attributes_path=vector_path, raw=raw)


def _validate_level_vectors(spec: Any, level_count: int) -> None:
    from zshar.data import load_activity_vectors

    vector_path = VECTOR_PATHS[level_count]
    vectors, attributes = load_activity_vectors(vector_path)
    expected_attributes = [
        str(item["name"]) for item in read_json(base.A1_ONTOLOGY).get("attributes", [])
    ]
    if set(vectors) != set(spec.activity_names):
        raise ValueError(f"{level_count}-level vectors do not cover PAMAP2 activities")
    if attributes != expected_attributes or len(attributes) != 9:
        raise ValueError(
            f"{level_count}-level vectors must keep the ordered A1 v1.5 9-D schema"
        )

    expected_levels = set(LEVEL_LABELS[level_count])
    observed_all: set[str] = set()
    for attribute in attributes:
        observed = {str(vector[attribute]) for vector in vectors.values()}
        observed_all.update(observed)
        if not observed or not observed.issubset(expected_levels):
            raise ValueError(
                f"{attribute} in {vector_path} contains labels outside the configured "
                f"{level_count}-level scale; observed={sorted(observed)}"
            )
    if observed_all != expected_levels:
        raise ValueError(
            f"{vector_path} must collectively cover the configured {level_count}-level "
            f"scale; observed={sorted(observed_all)}"
        )


def preflight(selected: Sequence[int], level_count: int) -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_dataset_table
    from zshar.splits import (
        load_sample_manifest,
        load_split_manifest,
        sample_evaluation_rows,
    )

    for path in (
        base.SPLITS,
        base.SAMPLES,
        base.A1_ONTOLOGY,
        VECTOR_PATHS[level_count],
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    split_payload = read_json(base.SPLITS)
    sample_payload = read_json(base.SAMPLES)
    if (
        split_payload.get("dataset") != base.DATASET
        or split_payload.get("evaluation_protocol") != "gzsl"
        or split_payload.get("training_label_space") != "seen_only"
        or split_payload.get("candidate_label_space") != "seen_union_unseen"
        or int(split_payload.get("seed", -1)) != 42
        or int(split_payload.get("num_splits", -1)) != 20
        or int(split_payload.get("unseen_count", -1))
        != base.matched.UNSEEN_ACTIVITY_COUNT
    ):
        raise ValueError(f"Invalid PAMAP2 seed-42 GZSL split manifest: {base.SPLITS}")
    if (
        sample_payload.get("dataset") != base.DATASET
        or sample_payload.get("evaluation_protocol") != "gzsl"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("seen_samples_per_activity", -1))
        != base.matched.EVAL_SAMPLES_PER_ACTIVITY
        or int(sample_payload.get("unseen_samples_per_activity", -1))
        != base.matched.EVAL_SAMPLES_PER_ACTIVITY
    ):
        raise ValueError(f"Invalid PAMAP2 GZSL eval20/seen20 manifest: {base.SAMPLES}")

    spec = dataset_spec(level_count)
    _validate_level_vectors(spec, level_count)
    splits = load_split_manifest(base.SPLITS, spec)
    indices = [int(split.index) for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    if set(map(int, sample_payload.get("samples", {}))) != set(indices):
        raise ValueError("GZSL sample-manifest split indices differ from split manifest")

    sample_manifest = load_sample_manifest(base.SAMPLES)
    table = load_dataset_table(spec)
    selected_set = set(map(int, selected))
    for split in splits:
        if split.index not in selected_set:
            continue
        by_activity = sample_payload["samples"][str(split.index)]
        if (
            len(split.seen_activities) != base.matched.SEEN_ACTIVITY_COUNT
            or len(split.unseen_activities) != base.matched.UNSEEN_ACTIVITY_COUNT
        ):
            raise ValueError(
                f"split {split.index} must contain {base.matched.SEEN_ACTIVITY_COUNT} seen + "
                f"{base.matched.UNSEEN_ACTIVITY_COUNT} unseen activities"
            )
        if set(map(str, by_activity)) != set(spec.activity_names):
            raise ValueError(f"split {split.index} samples do not cover all activities")
        if len(by_activity) != base.matched.TOTAL_ACTIVITY_COUNT or any(
            len(ids) != base.matched.EVAL_SAMPLES_PER_ACTIVITY
            or len(set(map(int, ids))) != base.matched.EVAL_SAMPLES_PER_ACTIVITY
            for ids in by_activity.values()
        ):
            raise ValueError(
                f"split {split.index} must have 20 fixed samples for all "
                f"{base.matched.TOTAL_ACTIVITY_COUNT} activities"
            )
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="gzsl",
            samples_per_activity=base.matched.EVAL_SAMPLES_PER_ACTIVITY,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        expected_rows = (
            base.matched.TOTAL_ACTIVITY_COUNT
            * base.matched.EVAL_SAMPLES_PER_ACTIVITY
        )
        if len(evaluation) != expected_rows or not evaluation.activity_name.value_counts().eq(
            base.matched.EVAL_SAMPLES_PER_ACTIVITY
        ).all():
            raise ValueError(
                f"split {split.index} does not resolve to "
                f"{base.matched.TOTAL_ACTIVITY_COUNT} x 20 balanced GZSL rows"
            )
        evaluation_ids = set(evaluation.window_index.astype(int))
        training = table.frame[
            table.frame.activity_name.isin(split.seen_activities)
            & ~table.frame.window_index.astype(int).isin(evaluation_ids)
        ]
        if set(training.activity_name.astype(str)) != set(split.seen_activities):
            raise ValueError(f"split {split.index} is missing outer-seen training rows")
        if set(training.window_index.astype(int)) & evaluation_ids:
            raise AssertionError(f"split {split.index} has train/test window leakage")

    print(
        f"Validated PAMAP2 {level_count}-level GZSL protocol for "
        f"{len(selected_set)} split(s); each split has "
        f"{base.matched.SEEN_ACTIVITY_COUNT} seen + "
        f"{base.matched.UNSEEN_ACTIVITY_COUNT} unseen activities and 20 fixed "
        "test rows per class.",
        flush=True,
    )
    return spec, splits, indices


def experiment_config(args: argparse.Namespace, selected: Sequence[int]) -> Any:
    configured = base.experiment_config(args, selected)
    return replace(configured, run_name=args.run_name)


def validate_existing_run(config: Any, spec: Any) -> Path:
    run_dir = base.validate_existing_run(config)
    saved_profile = run_dir / "dataset_profile.json"
    if saved_profile.is_file():
        recorded = Path(str(read_json(saved_profile).get("attributes", ""))).resolve()
        if recorded != spec.attributes_path.resolve():
            raise ValueError(
                f"Existing run uses different attribute vectors: {saved_profile}"
            )
    return run_dir


def summarize(
    config: Any,
    selected: Sequence[int],
    *,
    level_count: int,
) -> Path:
    run_dir = Path(config.run_root) / base.DATASET / str(config.run_name)
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(summary)

    frame = pd.read_csv(summary).sort_values("split_index")
    expected = set(map(int, selected))
    if len(frame) != len(expected) or set(frame.split_index.astype(int)) != expected:
        raise ValueError(f"Incomplete GZSL summary: {summary}")
    if set(frame.backend.astype(str)) != {"autogluon"}:
        raise ValueError(f"Unexpected backend in {summary}")
    if "attribute_order_path" in frame and frame.attribute_order_path.notna().any():
        raise ValueError("Independent run unexpectedly recorded an attribute order")
    if set(frame.gzsl_bias_correction.astype(str)) != {"temperature_seen_penalty"}:
        raise ValueError(f"Unexpected bias correction in {summary}")

    rows: list[dict[str, Any]] = []
    for metric in base.REPORT_METRICS:
        if metric not in frame:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "split_count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(run_dir / "gzsl_report.csv", index=False)
    vector_path = VECTOR_PATHS[level_count].resolve()
    write_json(
        run_dir / "gzsl_report.json",
        {
            "dataset": base.DATASET,
            "scenario": "gzsl",
            "backend": "autogluon",
            "classifier": "independent AutoGluon default ensemble per attribute",
            "attribute_set": f"initial_A1_v1_5_9d_{level_count}level",
            "attribute_level_count": level_count,
            "attribute_level_order": list(LEVEL_LABELS[level_count]),
            "attribute_vectors_path": str(vector_path),
            "attribute_vectors_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
            "attribute_dependency": "independent",
            "split_manifest": str(base.SPLITS),
            "sample_manifest": str(base.SAMPLES),
            "calibration": "3-fold outer-seen-only pseudo-GZSL grid calibration",
            "bias_correction": "temperature_seen_penalty",
            "split_count": len(expected),
            "metrics": rows,
        },
    )
    print("\n" + report.to_string(index=False), flush=True)
    print(f"GZSL report saved to: {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test", "report"), default="train-test"
    )
    parser.add_argument("--level-count", type=int, choices=tuple(LEVEL_LABELS), required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--split-indices", help="Optional comma-separated split subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id != 0:
        raise ValueError("This launcher is intentionally pinned to physical GPU 0")
    if args.num_cpus <= 0 or args.time_limit != 300:
        raise ValueError("This matched protocol requires positive CPUs and --time-limit 300")
    args.run_name = args.run_name or RUN_NAMES[args.level_count]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    split_payload = read_json(base.SPLITS)
    all_indices = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    selected = base.matched.selected_indices(args, all_indices)
    spec, _, verified = preflight(selected, args.level_count)
    if verified != all_indices:
        raise ValueError("GZSL split manifest changed during protocol validation")
    config = experiment_config(args, selected)
    run_dir = validate_existing_run(config, spec)

    print(
        f"PAMAP2 initial-A1 v1.5 {args.level_count}-level calibrated GZSL\n"
        "  attribute dependency: independent (no attribute order)\n"
        "  classifier: AutoGluon default model suite + weighted ensemble\n"
        "  matching: normalized ordinal probability\n"
        "  calibration: 3-fold outer-seen-only pseudo-GZSL (3 rows/activity)\n"
        "  bias correction: temperature softmax + calibrated seen-class penalty\n"
        "  physical GPU: 0 (logical device 0)\n"
        f"  split manifest: {base.SPLITS}\n"
        f"  sample manifest: {base.SAMPLES}\n"
        f"  selected splits: {selected}\n"
        f"  attribute vectors: {spec.attributes_path}\n"
        f"  level order: {list(LEVEL_LABELS[args.level_count])}\n"
        f"  run: {run_dir}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier or calibration model was trained.", flush=True)
        return 0
    if args.mode == "report":
        summarize(config, selected, level_count=args.level_count)
        return 0

    base.preserve_partial_models(run_dir, resume=args.resume)
    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, config)
    print(
        f"Completed PAMAP2 initial-A1 {args.level_count}-level calibrated GZSL: "
        f"{output}",
        flush=True,
    )
    summarize(config, selected, level_count=args.level_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
