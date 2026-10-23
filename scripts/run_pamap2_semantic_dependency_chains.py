#!/usr/bin/env python3
"""Build semantic-dependency orders and run PAMAP2 LR/CatBoost hard chains."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SPLITS = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
A2_ROOT = Path(
    "/home/wanzhen/LLMbasedZeroshotHARAgent/pamap2_pipeline_runs/"
    "pamap2_random20_unseen4_eval50_no_rerank_keep_models_seed42"
)
ORDER_DIR = PROJECT / "configs/attribute_orders/pamap2"
ORDER_FILE = ORDER_DIR / "semantic_dependency_dp_seed42.json"
COLLECTOR = PROJECT / "scripts/collect_pamap2_attribute_orders.py"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
RUN_SUFFIX = "random20_eval50_seed42"
BACKENDS = ("catboost_classifier_chain", "logistic_regression_chain")
VARIANTS = ("forward", "reverse")
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "attribute_exact_match",
)
MANAGER_MANIFEST = "attribute_backend_manifest.json"


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


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_vector(split_index: int) -> Path:
    candidates = (
        A2_ROOT / f"split_{split_index}/activity_attribute_vectors_a2.json",
        A2_ROOT / f"split_{split_index:03d}/activity_attribute_vectors_a2.json",
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not existing:
        raise FileNotFoundError(
            f"Missing old-pipeline A2 vectors for split {split_index}: "
            + " or ".join(map(str, candidates))
        )
    if len(existing) > 1 and len({sha256(path) for path in existing}) != 1:
        raise ValueError(f"Conflicting padded/unpadded A2 vectors for split {split_index}")
    return existing[0]


def preflight() -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table, replace_activity_vectors
    from zshar.datasets import load_dataset_spec
    from zshar.splits import (
        load_sample_manifest,
        load_split_manifest,
        sample_evaluation_rows,
    )

    for path in (SPLITS, SAMPLES, A2_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    split_payload = read_json(SPLITS)
    sample_payload = read_json(SAMPLES)
    if split_payload.get("dataset") != "pamap2" or int(split_payload.get("seed", -1)) != 42:
        raise ValueError(f"Expected PAMAP2 seed-42 split manifest: {SPLITS}")
    if (
        sample_payload.get("dataset") != "pamap2"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != 50
    ):
        raise ValueError(f"Expected PAMAP2 seed-42 eval50 sample manifest: {SAMPLES}")

    spec = load_dataset_spec("pamap2")
    splits = load_split_manifest(SPLITS, spec)
    indices = [split.index for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 splits")
    raw_samples = sample_payload.get("samples", {})
    if set(map(int, raw_samples)) != set(indices):
        raise ValueError("Sample-manifest split indices differ from split manifest")
    samples = load_sample_manifest(SAMPLES)
    base_table = load_dataset_table(spec)
    for split in splits:
        vector_path = resolve_vector(split.index)
        vectors, attributes = load_activity_vectors(vector_path)
        if set(vectors) != set(spec.activity_names):
            raise ValueError(f"Split {split.index} A2 activities differ from PAMAP2")
        if not attributes or len(attributes) != len(set(attributes)):
            raise ValueError(f"Split {split.index} has an invalid A2 schema")
        table = replace_activity_vectors(base_table, vector_path)
        by_activity = raw_samples[str(split.index)]
        if set(map(str, by_activity)) != set(split.unseen_activities):
            raise ValueError(f"Split {split.index} eval activities differ from unseen classes")
        if len(by_activity) != 4 or any(
            len(ids) != 50 or len(set(map(int, ids))) != 50
            for ids in by_activity.values()
        ):
            raise ValueError(f"Split {split.index} must have 4 unseen classes x 50 samples")
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=50,
            seed=42,
            manifest_ids=samples[split.index],
        )
        if len(evaluation) != 200 or not evaluation.activity_name.value_counts().eq(50).all():
            raise ValueError(f"Split {split.index} does not resolve to 200 balanced samples")
    return spec, splits, indices


def prepare_orders(splits: Sequence[Any]) -> Path:
    from zshar.data import load_activity_vectors
    from zshar.semantics.dependency_order import semantic_dependency_order

    entries: dict[str, Any] = {}
    for split in splits:
        vector_path = resolve_vector(split.index)
        vectors, attributes = load_activity_vectors(vector_path)
        seen = list(split.seen_activities)
        prototypes = pd.DataFrame(
            [[vectors[activity][attribute] for attribute in attributes] for activity in seen],
            index=seen,
            columns=attributes,
        )
        result = semantic_dependency_order(prototypes)
        result.update(
            {
                "split_index": split.index,
                "seen_activities": seen,
                "unseen_activities": list(split.unseen_activities),
                "seen_prototypes": {
                    activity: {
                        attribute: str(vectors[activity][attribute])
                        for attribute in attributes
                    }
                    for activity in seen
                },
                "attribute_vectors": str(vector_path.resolve()),
                "attribute_vectors_sha256": sha256(vector_path),
            }
        )
        entries[str(split.index)] = result
        print(
            f"split {split.index}: K={len(attributes)}, "
            f"J(forward)={result['forward_objective']:.6f}, "
            f"J(reverse)={result['reverse_objective']:.6f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "dataset": "pamap2",
        "attribute_set": "old_pipeline_A2_refined_per_split",
        "seed": 42,
        "selection_scope": (
            "seen activity semantic prototypes only; one equally weighted row per seen "
            "activity; no sensor samples, fitted classifier, unseen prototype, or test metric"
        ),
        "method": "directional normalized conditional-entropy reduction + exact subset DP",
        "directional_dependency": "D[i,j] = 1 - H(A_j | A_i) / H(A_j)",
        "objective": "J(pi) = sum_{p<q} D[pi_p,pi_q]",
        "optimizer": "subset dynamic programming, O(K^2 * 2^K)",
        "zero_entropy_target_policy": (
            "D[i,j]=0 when H(A_j)=0 because the target has no uncertainty to reduce"
        ),
        "tie_breaking": "lexicographically earliest order in the split A2 schema",
        "forward_definition": "global maximizer of J(pi)",
        "reverse_definition": "exact reversal of the optimal forward order",
        "split_manifest": str(SPLITS.resolve()),
        "split_manifest_sha256": sha256(SPLITS),
        "attribute_vectors_root": str(A2_ROOT.resolve()),
        "splits": entries,
    }
    write_json(ORDER_FILE, payload)
    subprocess.run([sys.executable, str(COLLECTOR)], cwd=PROJECT, check=True)
    print(f"Saved semantic dependency orders: {ORDER_FILE}", flush=True)
    return ORDER_FILE


def validate_order_file(splits: Sequence[Any]) -> None:
    from zshar.data import load_activity_vectors

    payload = read_json(ORDER_FILE)
    if (
        payload.get("dataset") != "pamap2"
        or int(payload.get("seed", -1)) != 42
        or payload.get("split_manifest_sha256") != sha256(SPLITS)
    ):
        raise ValueError(f"Invalid semantic dependency order file: {ORDER_FILE}")
    if set(map(int, payload.get("splits", {}))) != {split.index for split in splits}:
        raise ValueError("Semantic order split indices differ from the manifest")
    for split in splits:
        entry = payload["splits"][str(split.index)]
        vector_path = resolve_vector(split.index)
        _, attributes = load_activity_vectors(vector_path)
        forward = list(map(str, entry.get("forward", [])))
        reverse = list(map(str, entry.get("reverse", [])))
        if (
            entry.get("attribute_vectors_sha256") != sha256(vector_path)
            or len(forward) != len(set(forward))
            or set(forward) != set(attributes)
            or reverse != forward[::-1]
        ):
            raise ValueError(f"Invalid semantic order for split {split.index}")


def parse_selection(raw: str | None, all_indices: Sequence[int]) -> list[int]:
    if not raw:
        return list(all_indices)
    selected = sorted({int(value) for value in raw.split(",") if value.strip()})
    unknown = sorted(set(selected).difference(all_indices))
    if unknown:
        raise ValueError(f"Unknown --split-indices: {unknown}")
    return selected


def parse_backends(raw: str) -> list[str]:
    requested = [value.strip() for value in raw.split(",") if value.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--backends must be a nonempty, duplicate-free comma-separated list")
    unknown = sorted(set(requested).difference(BACKENDS))
    if unknown:
        raise ValueError(f"Unsupported --backends: {unknown}; choices={list(BACKENDS)}")
    return requested


def backend_config(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    common = {
        "random_state": 42,
        "consider_labels_correlation": True,
        "chain_feature_encoding": "ordinal",
    }
    if backend == "catboost_classifier_chain":
        return {
            **common,
            "n_jobs": args.num_cpus,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "task_type": "GPU",
            "devices": "0",
            "auto_class_weights": "Balanced",
            "allow_writing_files": False,
        }
    if backend == "logistic_regression_chain":
        return {
            **common,
            "max_iter": args.max_iter,
            "class_weight": "balanced",
            "C": args.c,
        }
    raise ValueError(backend)


def run_name(backend: str, variant: str) -> str:
    return f"pamap2_zsl_{backend}_a2_semantic_dependency_dp_{variant}_{RUN_SUFFIX}"


def experiment_config(
    args: argparse.Namespace,
    backend: str,
    variant: str,
    selected: Sequence[int],
):
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset="pamap2",
        scenario="zsl",
        backend=backend,
        backend_config=backend_config(args, backend),
        seed=42,
        split_manifest=str(SPLITS),
        sample_manifest=str(SAMPLES),
        attribute_vectors_root=str(A2_ROOT),
        attribute_order_config=str(ORDER_FILE),
        attribute_order_variant=variant,
        split_indices=set(selected),
        samples_per_activity=50,
        prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy",
        run_root=str(args.run_root.expanduser().resolve()),
        run_name=run_name(backend, variant),
        reuse_models=args.resume,
        evaluate_only=False,
        dry_run=args.dry_run,
    )


def canonical_config(config: Any) -> dict[str, Any]:
    payload = asdict(config)
    if payload.get("split_indices") is not None:
        payload["split_indices"] = sorted(map(int, payload["split_indices"]))
    return payload


def validate_existing_run(config: Any) -> Path:
    run_dir = Path(config.run_root) / "pamap2" / str(config.run_name)
    saved_config = run_dir / "config.json"
    if saved_config.is_file():
        previous = read_json(saved_config)
        if previous.get("split_indices") is not None:
            previous["split_indices"] = sorted(map(int, previous["split_indices"]))
        current = canonical_config(config)
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        if {key: value for key, value in previous.items() if key not in ignored} != {
            key: value for key, value in current.items() if key not in ignored
        }:
            raise ValueError(f"Existing run uses a different protocol: {saved_config}")
    return run_dir


def preserve_partial_models(run_dir: Path, *, resume: bool, dry_run: bool) -> None:
    partial = [
        model_dir
        for model_dir in run_dir.glob("split_*/models")
        if not (model_dir / MANAGER_MANIFEST).is_file()
    ]
    if not partial:
        return
    if not resume:
        raise ValueError(
            "Incomplete model directories exist; rerun with --resume: "
            + ", ".join(str(path) for path in partial)
        )
    if dry_run:
        for path in partial:
            print(f"Dry run: would preserve incomplete model directory {path}")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in partial:
        destination = path.with_name(f"models.incomplete_{stamp}")
        counter = 1
        while destination.exists():
            destination = path.with_name(f"models.incomplete_{stamp}_{counter}")
            counter += 1
        shutil.move(str(path), str(destination))
        print(f"Preserved incomplete model directory: {destination}", flush=True)


def summarize(
    args: argparse.Namespace,
    configs: Sequence[Any],
    selected: Sequence[int],
) -> Path:
    expected = set(selected)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for config in configs:
        key = (str(config.backend), str(config.attribute_order_variant))
        path = Path(config.run_root) / "pamap2" / str(config.run_name) / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        frame = pd.read_csv(path).set_index("split_index").sort_index()
        if frame.index.has_duplicates or set(map(int, frame.index)) != expected:
            raise ValueError(f"Expected exactly {len(expected)} completed splits: {path}")
        if set(frame["backend"].astype(str)) != {key[0]}:
            raise ValueError(f"Unexpected backend in {path}")
        if set(frame["attribute_order_variant"].astype(str)) != {key[1]}:
            raise ValueError(f"Unexpected order variant in {path}")
        frames[key] = frame
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            aggregate_rows.append(
                {
                    "backend": key[0],
                    "attribute_order": key[1],
                    "metric": metric,
                    "split_count": len(frame),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    paired_rows: list[pd.DataFrame] = []
    executed_backends = list(dict.fromkeys(config.backend for config in configs))
    for backend in executed_backends:
        forward = frames[(backend, "forward")]
        reverse = frames[(backend, "reverse")]
        for column in (
            "seen_activities",
            "unseen_activities",
            "eval_samples",
            "attribute_vectors_sha256",
            "attribute_order_sha256",
        ):
            if not forward[column].equals(reverse[column]):
                raise ValueError(f"{backend} forward/reverse differ in {column}")
        paired = pd.DataFrame(
            {"backend": backend, "split_index": forward.index}, index=forward.index
        )
        for metric in METRICS:
            paired[f"forward_{metric}"] = forward[metric]
            paired[f"reverse_{metric}"] = reverse[metric]
            paired[f"reverse_minus_forward_{metric}"] = reverse[metric] - forward[metric]
        paired_rows.append(paired.reset_index(drop=True))

    output = (
        args.run_root.expanduser().resolve()
        / "pamap2"
        / f"pamap2_zsl_semantic_dependency_dp_chain_comparison_{RUN_SUFFIX}"
    )
    output.mkdir(parents=True, exist_ok=True)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "comparison.csv", index=False)
    pd.concat(paired_rows, ignore_index=True).to_csv(
        output / "paired_split_metrics.csv", index=False
    )
    write_json(
        output / "comparison.json",
        {
            "dataset": "pamap2",
            "scenario": "zsl",
            "attribute_set": "old_pipeline_A2_refined_per_split",
            "order_method": "directional normalized conditional entropy + exact subset DP",
            "backends": executed_backends,
            "order_sequence": list(VARIANTS),
            "split_count_per_configuration": len(expected),
            "attribute_order_config": str(ORDER_FILE.resolve()),
            "attribute_order_sha256": sha256(ORDER_FILE),
            "metrics": aggregate_rows,
        },
    )
    print("\n" + aggregate.to_string(index=False), flush=True)
    print(f"Combined comparison saved to: {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("check", "prepare", "train-test", "report"),
        default="train-test",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--split-indices", help="Optional comma-separated training subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--num-cpus", type=int, default=112)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_id < 0 or args.num_cpus <= 0:
        raise ValueError("--gpu-id must be nonnegative and --num-cpus must be positive")
    if min(args.iterations, args.depth, args.max_iter) <= 0:
        raise ValueError("Iteration/depth settings must be positive")
    if args.learning_rate <= 0 or args.c <= 0:
        raise ValueError("--learning-rate and --c must be positive")
    requested_backends = parse_backends(args.backends)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    spec, splits, all_indices = preflight()
    selected = parse_selection(args.split_indices, all_indices)
    print(
        "PAMAP2 semantic-dependency chain protocol\n"
        "  ordering input: seen activity A2 semantic prototypes only\n"
        "  optimizer: exact subset DP for directional conditional-entropy reduction\n"
        f"  chain backends: {requested_backends}\n"
        "  variants per backend: forward -> exact reverse\n"
        f"  selected training/evaluation splits: {selected}\n"
        f"  CatBoost physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  order file: {ORDER_FILE}",
        flush=True,
    )
    if args.mode == "check":
        return 0
    if args.mode in {"prepare", "train-test"}:
        prepare_orders(splits)
    elif not ORDER_FILE.is_file():
        raise FileNotFoundError(f"Prepare semantic dependency orders first: {ORDER_FILE}")
    validate_order_file(splits)
    if args.mode == "prepare":
        return 0

    configs = [
        experiment_config(args, backend, variant, selected)
        for backend in requested_backends
        for variant in VARIANTS
    ]
    run_dirs = [validate_existing_run(config) for config in configs]
    if args.mode == "report":
        summarize(args, configs, selected)
        return 0
    for run_dir in run_dirs:
        preserve_partial_models(run_dir, resume=args.resume, dry_run=args.dry_run)
    from zshar.pipeline.runner import run_experiment

    for position, (config, run_dir) in enumerate(zip(configs, run_dirs), 1):
        print(
            f"\nStage {position}/{len(configs)}: backend={config.backend}, "
            f"order={config.attribute_order_variant}\n  run={run_dir}",
            flush=True,
        )
        run_experiment(spec, config)
    if not args.dry_run:
        summarize(args, configs, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
