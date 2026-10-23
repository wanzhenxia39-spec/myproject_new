#!/usr/bin/env python3
"""Run latest MHEALTH A2 hard-label AutoGluon chains in reverse and forward order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SPLITS = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/mhealth_unseen4_random20_eval30_samples_seed42.json"
A2_ROOT = PROJECT / "runtime/refinement_planner1_kami/mhealth_v1_5"
OLD_ORDERS = Path("/home/wanzhen/myproject/configs/attributes/mhealth_atrribute_order.json")
ORDER_FILE = A2_ROOT / "mhealth_a2_attribute_orders_seed42.json"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
VARIANTS = ("reverse", "forward")
METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
)
MANAGER_MANIFEST = "attribute_backend_manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_indices() -> list[int]:
    payload = read_json(SPLITS)
    if payload.get("dataset") != "mhealth" or int(payload.get("seed", -1)) != 42:
        raise ValueError(f"Expected the MHEALTH seed-42 split manifest: {SPLITS}")
    indices = [int(row["split_index"]) for row in payload.get("splits", [])]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError(f"Expected 20 distinct split indices, found {indices}")
    return indices


def vector_path(split_index: int) -> Path:
    return A2_ROOT / f"split_{split_index}" / "activity_attribute_vectors_a2.json"


def plan_path(split_index: int) -> Path:
    return A2_ROOT / f"split_{split_index}" / "a2_attribute_plan.json"


def incomplete_splits(indices: Sequence[int]) -> list[int]:
    missing = []
    for index in indices:
        if not vector_path(index).is_file() or not plan_path(index).is_file():
            missing.append(index)
            continue
        try:
            vectors = read_json(vector_path(index))
            plan = read_json(plan_path(index))
            if not isinstance(vectors, Mapping) or not vectors:
                raise ValueError("empty vectors")
            rows = list(vectors.values())
            attributes = list(rows[0])
            if not attributes or any(
                not isinstance(row, Mapping)
                or list(row) != attributes
                or any(value not in {"Low", "Medium", "High"} for value in row.values())
                for row in rows
            ):
                raise ValueError("invalid vector rows")
            plan_attributes = [item["name"] for item in plan.get("attributes", [])]
            if plan_attributes != attributes:
                raise ValueError("plan/vector attribute mismatch")
            decision = (plan.get("_meta") or {}).get("binding_decision")
            if decision not in {"validated_a2", "fallback_a1"}:
                raise ValueError("plan has no final binding decision")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            missing.append(index)
    return missing


def wait_for_vectors(indices: Sequence[int], *, wait: bool, poll_seconds: float) -> None:
    while True:
        missing = incomplete_splits(indices)
        if not missing:
            return
        if not wait:
            raise RuntimeError(
                "Latest A2 plans/vectors are not complete for all 20 splits. "
                f"Pending/invalid splits: {missing}. Rerun with --wait-for-vectors to wait."
            )
        print(
            f"Waiting for latest A2 vectors: {20 - len(missing)}/20 complete; "
            f"pending={missing}",
            flush=True,
        )
        time.sleep(poll_seconds)


def validate_samples(indices: Sequence[int]) -> None:
    payload = read_json(SAMPLES)
    split_rows = {
        int(row["split_index"]): set(map(str, row["unseen_activities"]))
        for row in read_json(SPLITS)["splits"]
    }
    if (
        payload.get("dataset") != "mhealth"
        or int(payload.get("seed", -1)) != 42
        or int(payload.get("samples_per_activity", -1)) != 30
    ):
        raise ValueError(f"Expected fixed MHEALTH seed-42 eval30 samples: {SAMPLES}")
    samples = payload.get("samples", {})
    if set(map(int, samples)) != set(indices):
        raise ValueError("Sample manifest split indices do not match the split manifest")
    for index in indices:
        activity_samples = samples[str(index)]
        if set(map(str, activity_samples)) != split_rows[index]:
            raise ValueError(
                f"Split {index} sample activities do not match its unseen activities"
            )
        if len(activity_samples) != 4 or any(
            len(ids) != 30 or len(set(map(int, ids))) != 30
            for ids in activity_samples.values()
        ):
            raise ValueError(
                f"Split {index} must contain four activities with 30 unique samples each"
            )


def build_compatible_orders(indices: Sequence[int]) -> dict[str, Any]:
    """Project the saved old priority order onto each latest A2 schema.

    Attributes retained by the new schema keep their relative position from the
    old forward order. New attributes, which have no saved priority, follow in
    the deterministic order stored in the latest A2 plan. Reverse is the exact
    reversal of this complete compatible forward order.
    """
    old = read_json(OLD_ORDERS)
    if old.get("dataset") != "mhealth" or int(old.get("seed", -1)) != 42:
        raise ValueError(f"Saved order source is not MHEALTH seed42: {OLD_ORDERS}")

    entries: dict[str, Any] = {}
    vector_hashes: dict[str, str] = {}
    exact_reuse_count = 0
    for index in indices:
        plan = read_json(plan_path(index))
        attributes = [str(item["name"]) for item in plan["attributes"]]
        previous = old.get("splits", {}).get(str(index))
        if not isinstance(previous, Mapping):
            raise ValueError(f"Saved order source has no split {index}")
        old_forward = list(map(str, previous.get("forward", [])))
        old_reverse = list(map(str, previous.get("reverse", [])))
        if not old_forward or old_reverse != old_forward[::-1]:
            raise ValueError(f"Saved split {index} reverse is not an exact reversal")
        retained = [name for name in old_forward if name in attributes]
        added = [name for name in attributes if name not in retained]
        forward = retained + added
        if len(forward) != len(set(forward)) or set(forward) != set(attributes):
            raise AssertionError(f"Failed to construct a permutation for split {index}")
        exact = forward == old_forward
        exact_reuse_count += int(exact)
        entries[str(index)] = {
            "forward": forward,
            "reverse": forward[::-1],
            "reuse_mode": "exact" if exact else "projected_to_latest_a2",
            "retained_from_saved_forward": retained,
            "new_attributes_appended_in_a2_order": added,
            "removed_old_attributes": [name for name in old_forward if name not in attributes],
            "latest_a2_plan": str(plan_path(index).resolve()),
            "latest_a2_vectors": str(vector_path(index).resolve()),
            "latest_a2_vectors_sha256": file_hash(vector_path(index)),
        }
        vector_hashes[str(index)] = file_hash(vector_path(index))

    return {
        "schema_version": 1,
        "dataset": "mhealth",
        "seed": 42,
        "attribute_set": "latest_planner1_A2_refined_per_split",
        "selection_scope": "saved MHEALTH seed42 order projected without test data",
        "source_order_config": str(OLD_ORDERS.resolve()),
        "source_order_sha256": file_hash(OLD_ORDERS),
        "split_manifest": str(SPLITS.resolve()),
        "split_manifest_sha256": file_hash(SPLITS),
        "sample_manifest": str(SAMPLES.resolve()),
        "sample_manifest_sha256": file_hash(SAMPLES),
        "attribute_vectors_root": str(A2_ROOT.resolve()),
        "attribute_vector_sha256": vector_hashes,
        "forward_definition": (
            "retain the relative order of attributes present in the saved forward order, "
            "then append new latest-A2 attributes in A2 plan order"
        ),
        "reverse_definition": "exact full reversal of the compatible forward order",
        "exactly_reused_split_count": exact_reuse_count,
        "generated_at": timestamp(),
        "splits": entries,
    }


def same_order_inputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "dataset",
        "seed",
        "source_order_sha256",
        "split_manifest_sha256",
        "sample_manifest_sha256",
        "attribute_vector_sha256",
        "forward_definition",
        "reverse_definition",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def prepare_orders(indices: Sequence[int], *, rebuild: bool) -> Path:
    expected = build_compatible_orders(indices)
    if ORDER_FILE.is_file() and not rebuild:
        existing = read_json(ORDER_FILE)
        if not isinstance(existing, Mapping) or not same_order_inputs(existing, expected):
            raise ValueError(
                f"Existing compatible order has different inputs: {ORDER_FILE}. "
                "Use --rebuild-orders after confirming the A2 changes."
            )
        print(f"Reusing compatible latest-A2 order file: {ORDER_FILE}", flush=True)
        return ORDER_FILE
    if ORDER_FILE.is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = ORDER_FILE.with_name(f"{ORDER_FILE.name}.backup_{stamp}")
        shutil.copy2(ORDER_FILE, backup)
        print(f"Preserved previous compatible order file: {backup}", flush=True)
    write_json(ORDER_FILE, expected)
    print(f"Saved compatible forward/reverse orders: {ORDER_FILE}", flush=True)
    for index in indices:
        item = expected["splits"][str(index)]
        print(
            f"  split {index}: retained={len(item['retained_from_saved_forward'])}, "
            f"new={len(item['new_attributes_appended_in_a2_order'])}, "
            f"removed={len(item['removed_old_attributes'])}",
            flush=True,
        )
    return ORDER_FILE


def run_name(variant: str) -> str:
    return (
        "mhealth_zsl_autogluon_multilabel_hard_a2_"
        f"{variant}_random20_eval30_seed42"
    )


def runner_command(args: argparse.Namespace, variant: str, mode: str) -> list[str]:
    backend_config = json.dumps(
        {
            "consider_labels_correlation": True,
            "lightgbm_device": "cuda",
            "verbosity": args.verbosity,
        },
        separators=(",", ":"),
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "zshar.cli",
        "run",
        "--dataset",
        "mhealth",
        "--dataset-config",
        str(PROJECT / "configs/datasets/mhealth.json"),
        "--scenario",
        "zsl",
        "--backend",
        "autogluon_multilabel",
        "--backend-config",
        backend_config,
        "--seed",
        "42",
        "--split-manifest",
        str(SPLITS),
        "--sample-manifest",
        str(SAMPLES),
        "--attribute-vectors-root",
        str(A2_ROOT),
        "--attribute-order-config",
        str(ORDER_FILE),
        "--attribute-order-variant",
        variant,
        "--samples-per-activity",
        "30",
        "--prototype-metric",
        "ordinal_probability",
        "--attribute-weighting",
        "validation_accuracy",
        "--time-limit",
        str(args.time_limit),
        "--presets",
        args.presets,
        "--autogluon-model-suite",
        args.model_suite,
        "--num-cpus",
        str(args.num_cpus),
        "--num-gpus",
        "1",
        "--gpu-id",
        "1",
        "--num-bag-folds",
        "0",
        "--num-stack-levels",
        "0",
        "--run-root",
        str(args.run_root),
        "--run-name",
        run_name(variant),
    ]
    if mode == "evaluate":
        command += ["--evaluate-only", "--reuse-models"]
    elif args.resume:
        command.append("--reuse-models")
    return command


def preserve_partial_models(args: argparse.Namespace, variant: str) -> None:
    directory = args.run_root / "mhealth" / run_name(variant)
    for model_dir in directory.glob("split_*/models"):
        if (model_dir / MANAGER_MANIFEST).is_file():
            if not args.resume:
                raise FileExistsError(
                    f"Completed model exists: {model_dir}. Use --resume or a new --run-root."
                )
            continue
        if not args.resume:
            raise FileExistsError(
                f"Partial model directory exists: {model_dir}. Use --resume to preserve and retrain it."
            )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = model_dir.with_name(f"models.incomplete_{stamp}")
        counter = 1
        while backup.exists():
            backup = model_dir.with_name(f"models.incomplete_{stamp}_{counter}")
            counter += 1
        shutil.move(str(model_dir), str(backup))
        print(f"Preserved partial model directory: {backup}", flush=True)


def execute(command: Sequence[str]) -> None:
    print("$ " + shlex.join(command), flush=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(PROJECT / "src")
        + (os.pathsep + os.environ["PYTHONPATH"] if os.getenv("PYTHONPATH") else ""),
    }
    subprocess.run(list(command), cwd=PROJECT, env=environment, check=True)


def summarize(args: argparse.Namespace, indices: Sequence[int]) -> Path:
    frames: dict[str, pd.DataFrame] = {}
    expected = set(indices)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        directory = args.run_root / "mhealth" / run_name(variant)
        summary_path = directory / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing completed summary: {summary_path}")
        frame = pd.read_csv(summary_path).set_index("split_index").sort_index()
        if frame.index.has_duplicates or set(map(int, frame.index)) != expected:
            raise ValueError(f"Expected exactly the fixed 20 splits in {summary_path}")
        if set(frame["backend"].astype(str)) != {"autogluon_multilabel"}:
            raise ValueError(f"Unexpected backend in {summary_path}")
        if set(frame["attribute_order_variant"].astype(str)) != {variant}:
            raise ValueError(f"Unexpected order variant in {summary_path}")
        frames[variant] = frame
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            rows.append(
                {
                    "attribute_order": variant,
                    "metric": metric,
                    "split_count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    forward, reverse = frames["forward"], frames["reverse"]
    for column in (
        "seen_activities",
        "unseen_activities",
        "eval_samples",
        "attribute_vectors_sha256",
        "attribute_order_sha256",
    ):
        if not forward[column].equals(reverse[column]):
            raise ValueError(f"Forward/reverse protocols differ in {column}")

    output = (
        args.run_root
        / "mhealth"
        / "mhealth_hard_chain_forward_reverse_a2_seed42"
    )
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    write_json(
        output / "comparison.json",
        {
            "dataset": "mhealth",
            "scenario": "zsl",
            "backend": "autogluon_multilabel",
            "hard_labels_chained": True,
            "split_count_per_order": 20,
            "order_sequence": list(VARIANTS),
            "metrics": rows,
        },
    )
    print("\n" + comparison.to_string(index=False), flush=True)
    print(f"Comparison saved to: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", "train-test", "evaluate", "report"),
        help="Prepare compatible orders, train+test both orders, reevaluate saved models, or report.",
    )
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--time-limit", type=int, default=600, help="Seconds per attribute predictor")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--model-suite", choices=("default", "zshar_five"), default="default")
    parser.add_argument("--verbosity", type=int, choices=range(5), default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-vectors", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--rebuild-orders", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and print both commands without training or evaluation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root = args.run_root.expanduser().resolve()
    if min(args.time_limit, args.num_cpus) <= 0 or args.poll_seconds <= 0:
        raise ValueError("Time limit, CPU count, and poll interval must be positive")
    for path in (SPLITS, SAMPLES, OLD_ORDERS):
        if not path.is_file():
            raise FileNotFoundError(path)

    indices = split_indices()
    validate_samples(indices)
    wait_for_vectors(indices, wait=args.wait_for_vectors, poll_seconds=args.poll_seconds)
    prepare_orders(indices, rebuild=args.rebuild_orders)

    print(
        "MHEALTH strict ZSL: 20 fixed seed42 splits; 4 unseen activities x 30 samples.\n"
        "Backend: AutoGluon hard-label multilabel classifier chain.\n"
        "Execution order: reverse 20 splits, then forward 20 splits.\n"
        "Physical GPU: 1 (visible as CUDA device 0 inside each worker).",
        flush=True,
    )
    if args.mode == "prepare":
        return 0
    if args.mode == "report":
        summarize(args, indices)
        return 0

    for variant in VARIANTS:
        command = runner_command(args, variant, args.mode)
        if args.dry_run:
            print("$ " + shlex.join(command))
            continue
        if args.mode == "train-test":
            preserve_partial_models(args, variant)
        execute(command)
    if not args.dry_run:
        summarize(args, indices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
