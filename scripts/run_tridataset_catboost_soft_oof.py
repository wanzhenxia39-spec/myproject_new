#!/usr/bin/env python3
"""Run PAMAP2, MHEALTH and WISDM CatBoost soft-OOF chains sequentially."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT = Path("/home/wanzhen/myproject_new")
NEW_SRC = PROJECT / "src"
REQUESTED_BACKEND = Path(
    "/home/wanzhen/myproject/src/zshar/models/catboost_soft_oof_backend.py"
)
ENTRYPOINT = PROJECT / "scripts/run_zshar_with_myproject_catboost_soft_oof.py"
RUN_ROOT = PROJECT / "runtime/zshar_runs"
MANAGER_MANIFEST = "attribute_backend_manifest.json"
DATASETS: dict[str, dict[str, Any]] = {
    "pamap2": {"unseen": 4, "samples": 50},
    "mhealth": {"unseen": 4, "samples": 30},
    "wisdm": {"unseen": 6, "samples": 30},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def paths(dataset: str, profile: str) -> dict[str, Path]:
    unseen = DATASETS[dataset]["unseen"]
    samples = DATASETS[dataset]["samples"]
    result = {
        "dataset_config": PROJECT / f"configs/datasets/{dataset}.json",
        "splits": PROJECT
        / f"split/zero_shot_split/{dataset}_unseen{unseen}_random20_splits_seed42.json",
        "samples": PROJECT
        / (
            f"split/zero_shot_split/{dataset}_unseen{unseen}_random20_"
            f"eval{samples}_samples_seed42.json"
        ),
    }
    if profile == "a2":
        result["vectors"] = (
            PROJECT / f"runtime/refinement_semantic_v3/{dataset}_gpt_5_6_sol_11_15"
        )
        result["orders"] = PROJECT / (
            f"runtime/{dataset}_a2_semantic_v3_catboost_seen_only_order_seed42/"
            "order_preparation/attribute_orders_catboost.json"
        )
    else:
        result["vectors"] = (
            PROJECT / f"configs/attributes/activity_attribute_vectors_initial_{dataset}_v1_5.json"
        )
        result["orders"] = PROJECT / (
            f"configs/attribute_orders/{dataset}/"
            "initial_a1_v1_5_catboost_seen_only_seed42.json"
        )
    return result


def run_name(dataset: str, profile: str, gpu_id: int) -> str:
    samples = DATASETS[dataset]["samples"]
    attribute_tag = "a2_semantic_v3" if profile == "a2" else "a1_v1_5"
    return (
        f"{dataset}_zsl_catboost_soft_oof_{attribute_tag}_catboost_seen_only_"
        f"forward_oof5_random20_eval{samples}_seed42_gpu{gpu_id}"
    )


def backend_config(num_cpus: int) -> dict[str, Any]:
    return {
        "random_state": 42,
        "n_jobs": num_cpus,
        "task_type": "GPU",
        "devices": "0",
        "allow_writing_files": False,
        "consider_labels_correlation": True,
        "oof_folds": 5,
        "oof_shuffle": True,
        "oof_random_state": 42,
        "uncertainty_metric": "normalized_entropy",
    }


def verify_requested_backend() -> None:
    module_name = "zshar.models._catboost_soft_oof_preflight"
    spec = importlib.util.spec_from_file_location(module_name, REQUESTED_BACKEND)
    if spec is None or spec.loader is None:
        raise ImportError(REQUESTED_BACKEND)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    actual = Path(inspect.getfile(module.CatBoostSoftOOFModelManager)).resolve()
    if actual != REQUESTED_BACKEND.resolve():
        raise RuntimeError(f"Unexpected backend source: {actual}")


def preflight(profile: str) -> None:
    if str(NEW_SRC) not in sys.path:
        sys.path.insert(0, str(NEW_SRC))
    verify_requested_backend()
    from zshar.data import load_activity_vectors
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_split_manifest

    for dataset, settings in DATASETS.items():
        current = paths(dataset, profile)
        for path in current.values():
            if not path.exists():
                raise FileNotFoundError(path)
        spec = load_dataset_spec(dataset, current["dataset_config"])
        splits = load_split_manifest(current["splits"], spec)
        indices = [int(split.index) for split in splits]
        if len(indices) != 20 or len(set(indices)) != 20:
            raise ValueError(f"{dataset}: expected exactly 20 seed-42 splits")
        split_payload = read_json(current["splits"])
        sample_payload = read_json(current["samples"])
        if split_payload.get("dataset") != dataset or int(split_payload.get("seed", -1)) != 42:
            raise ValueError(f"{dataset}: invalid split manifest")
        if (
            sample_payload.get("dataset") != dataset
            or int(sample_payload.get("seed", -1)) != 42
            or int(sample_payload.get("samples_per_activity", -1)) != settings["samples"]
            or set(map(int, sample_payload.get("samples", {}))) != set(indices)
        ):
            raise ValueError(f"{dataset}: invalid fixed-sample manifest")

        order_payload = read_json(current["orders"])
        if (
            order_payload.get("dataset") != dataset
            or int(order_payload.get("seed", -1)) != 42
            or order_payload.get("ranking_backend") != "catboost"
        ):
            raise ValueError(f"{dataset}: invalid CatBoost seen-only order metadata")
        entries = order_payload.get("splits", {})
        if set(map(int, entries)) != set(indices):
            raise ValueError(f"{dataset}: order file does not cover the exact 20 splits")

        if profile == "a1":
            vectors, attributes = load_activity_vectors(current["vectors"])
            if spec.attributes_path.resolve() != current["vectors"].resolve():
                raise ValueError(f"{dataset}: dataset profile does not point to requested A1")
            if len(attributes) != 9 or set(vectors) != set(spec.activity_names):
                raise ValueError(f"{dataset}: invalid initial A1 vectors")
            schemas = {index: attributes for index in indices}
        else:
            schemas = {}
            for index in indices:
                vector_file = (
                    current["vectors"]
                    / f"split_{index}"
                    / "activity_attribute_vectors_a2.json"
                )
                vectors, attributes = load_activity_vectors(vector_file)
                if not 11 <= len(attributes) <= 15 or set(vectors) != set(spec.activity_names):
                    raise ValueError(f"{dataset}/split {index}: invalid refined A2 vectors")
                schemas[index] = attributes
        for index in indices:
            forward = list(map(str, entries[str(index)].get("forward", [])))
            if len(forward) != len(set(forward)) or set(forward) != set(schemas[index]):
                raise ValueError(f"{dataset}/split {index}: stale forward order")
        print(
            f"Validated {dataset}: profile={profile.upper()}, splits=20, "
            f"fixed_eval={settings['unseen']}x{settings['samples']}, orders=20",
            flush=True,
        )


def complete(run_dir: Path) -> bool:
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        return False
    try:
        frame = pd.read_csv(summary)
    except pd.errors.EmptyDataError:
        return False
    if len(frame) != 20 or frame["split_index"].nunique() != 20:
        return False
    return all(
        (run_dir / f"split_{int(index):03d}" / "models" / MANAGER_MANIFEST).is_file()
        for index in frame["split_index"]
    )


def preserve_incomplete_models(run_dir: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for model_dir in run_dir.glob("split_*/models"):
        if (model_dir / MANAGER_MANIFEST).is_file():
            continue
        destination = model_dir.with_name(f"models_incomplete_{stamp}")
        counter = 1
        while destination.exists():
            destination = model_dir.with_name(f"models_incomplete_{stamp}_{counter}")
            counter += 1
        shutil.move(str(model_dir), str(destination))
        print(f"Preserved incomplete model directory: {destination}", flush=True)


def command(dataset: str, profile: str, gpu_id: int, num_cpus: int) -> list[str]:
    current = paths(dataset, profile)
    name = run_name(dataset, profile, gpu_id)
    result = [
        sys.executable,
        "-u",
        str(ENTRYPOINT),
        "run",
        "--dataset",
        dataset,
        "--dataset-config",
        str(current["dataset_config"]),
        "--scenario",
        "zsl",
        "--backend",
        "catboost_soft_oof",
        "--backend-config",
        json.dumps(backend_config(num_cpus), separators=(",", ":")),
        "--gpu-id",
        str(gpu_id),
        "--seed",
        "42",
        "--split-manifest",
        str(current["splits"]),
        "--sample-manifest",
        str(current["samples"]),
        "--attribute-order-config",
        str(current["orders"]),
        "--attribute-order-variant",
        "forward",
        "--samples-per-activity",
        str(DATASETS[dataset]["samples"]),
        "--prototype-metric",
        "ordinal_probability",
        "--attribute-weighting",
        "validation_accuracy",
        "--run-root",
        str(RUN_ROOT),
        "--run-name",
        name,
        "--reuse-models",
    ]
    if profile == "a2":
        result += ["--attribute-vectors-root", str(current["vectors"])]
    return result


def summarize(profile: str, gpu_id: int) -> None:
    rows = []
    for dataset in DATASETS:
        run_dir = RUN_ROOT / dataset / run_name(dataset, profile, gpu_id)
        frame = pd.read_csv(run_dir / "summary.csv")
        rows.append(
            {
                "dataset": dataset,
                "split_count": len(frame),
                "activity_top1_accuracy": frame["activity_top1_accuracy"].mean(),
                "activity_top3_recall": frame["activity_top3_recall"].mean(),
                "activity_macro_f1": frame["activity_macro_f1"].mean(),
                "attribute_exact_match": frame["attribute_exact_match"].mean(),
            }
        )
    output = RUN_ROOT / f"catboost_soft_oof_{profile}_tridataset_gpu{gpu_id}_seed42.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Saved three-dataset summary: {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train-test", "report"))
    parser.add_argument("--profile", choices=("a1", "a2"), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--num-cpus", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Keep the two attribute profiles on separate physical GPUs.
    expected_gpu = 0 if args.profile == "a2" else 1
    if args.gpu_id != expected_gpu:
        raise ValueError(f"Profile {args.profile} is reserved for physical GPU {expected_gpu}")
    if args.num_cpus <= 0:
        raise ValueError("--num-cpus must be positive")
    preflight(args.profile)
    if args.mode == "check":
        print("Check only: no CatBoost soft-OOF model was trained.", flush=True)
        return 0
    if args.mode == "report":
        summarize(args.profile, args.gpu_id)
        return 0

    environment = {
        **os.environ,
        "PYTHONPATH": str(NEW_SRC),
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
    }
    for position, dataset in enumerate(DATASETS, 1):
        run_dir = RUN_ROOT / dataset / run_name(dataset, args.profile, args.gpu_id)
        print(f"Stage {position}/3: {dataset.upper()} -> {run_dir}", flush=True)
        if complete(run_dir):
            print(f"Skipping completed dataset: {dataset}", flush=True)
            continue
        preserve_incomplete_models(run_dir)
        subprocess.run(
            command(dataset, args.profile, args.gpu_id, args.num_cpus),
            cwd=PROJECT,
            env=environment,
            check=True,
        )
        if not complete(run_dir):
            raise RuntimeError(f"{dataset} exited without a complete 20-split result")
    summarize(args.profile, args.gpu_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
