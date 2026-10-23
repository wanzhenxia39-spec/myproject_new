#!/usr/bin/env python3
"""Train/test all 13 non-chain, non-default-ensemble backends on v4 seed-42 ZSL.

PAMAP2 defaults to physical GPU 0; MHEALTH defaults to physical GPU 1.
Only the requested dataset is run. No refinement or migration job is started.
All writable experiment paths must resolve onto the /data data filesystem.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))
import attributes_v4_autogluon_common as v4

DATA_ROOT = Path("/data/wanzhen")
DEFAULT_OUTPUT = DATA_ROOT / "experiments/pamap2_mhealth_v4_nonchain_seed42"
BACKENDS = (
    "lightgbm", "xgboost", "catboost", "logistic_regression", "linear_svm",
    "extra_trees", "random_forest", "neuralnet_fastai", "neuralnet_torch",
    "ordinal_logistic", "coral_mlp", "mlp_independent", "mlp_multitask",
)
NEURAL = {
    "neuralnet_fastai", "neuralnet_torch", "ordinal_logistic", "coral_mlp",
    "mlp_independent", "mlp_multitask",
}
MODULES = {
    "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
    **{name: "torch" for name in NEURAL},
}
METRICS = (
    "activity_top1_accuracy", "activity_top3_recall", "activity_macro_f1",
    "attribute_exact_match",
)


def data_path(path: Path) -> Path:
    """Reject system-disk paths, including symlinks and nested other mounts."""
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError(f"Output must be under /data/wanzhen: {resolved}")
    ancestor = resolved
    while not ancestor.exists():
        ancestor = ancestor.parent
    device = Path("/data").stat().st_dev
    if device == Path("/").stat().st_dev or ancestor.stat().st_dev != device:
        raise ValueError(f"Output is not on the /data data filesystem: {resolved}")
    return resolved


def prepare_environment(args):
    args.output_root = data_path(args.output_root)
    # Check before mkdir, then verify again so nested symlinks cannot escape.
    runtime = data_path(args.output_root / args.dataset / "runtime")
    cache_variables = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "JOBLIB_TEMP_FOLDER": "joblib", "XDG_CACHE_HOME": "cache",
        "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib",
        "NUMBA_CACHE_DIR": "numba", "CUDA_CACHE_PATH": "cuda",
        "TRITON_CACHE_DIR": "triton", "HF_HOME": "huggingface",
        "PYTHONPYCACHEPREFIX": "pycache", "ZSHAR_RUNTIME_ROOT": "zshar",
    }
    for key, directory in cache_variables.items():
        path = data_path(runtime / directory)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(data_path(path))
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = str(PROJECT / "src") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.getenv("PYTHONPATH") else ""
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.num_cpus)


def backend_config(backend, num_cpus):
    config = {"random_state": 42}
    if backend == "lightgbm":
        config.update(n_jobs=num_cpus, device="cuda", gpu_device_id=0)
    elif backend == "xgboost":
        config.update(n_jobs=num_cpus, tree_method="hist", device="cuda:0")
    elif backend == "catboost":
        config.update(n_jobs=num_cpus, task_type="GPU", devices="0", allow_writing_files=False)
    elif backend in {"extra_trees", "random_forest"}:
        config["n_jobs"] = num_cpus
    elif backend == "linear_svm":
        config["calibrate_proba"] = True
    elif backend in NEURAL:
        config["device"] = "cuda:0"
    elif backend != "logistic_regression":
        raise ValueError(f"Unsupported backend: {backend}")
    return config


def run_name(dataset, backend):
    return f"{dataset}_zsl_{backend}_a2_v4_nonchain_random20_eval{v4.EVAL_COUNTS[dataset]}_seed42"


def experiment_config(args, backend):
    from zshar.pipeline.runner import ExperimentConfig

    split_path, sample_path = v4.manifest_paths(args.dataset)
    return ExperimentConfig(
        dataset=args.dataset, scenario="zsl", backend=backend,
        backend_config=backend_config(backend, args.num_cpus), seed=42,
        split_manifest=str(split_path), sample_manifest=str(sample_path),
        attribute_vectors_root=str(PROJECT / "configs/attrIbutes_v4" / args.dataset),
        samples_per_activity=v4.EVAL_COUNTS[args.dataset],
        prototype_metric="ordinal_probability", attribute_weighting="validation_accuracy",
        run_root=str(args.output_root), run_name=run_name(args.dataset, backend),
        reuse_models=args.resume or args.mode == "evaluate", evaluate_only=args.mode == "evaluate",
    )


def command(args, config):
    result = [sys.executable, "-u", "-m", "zshar.cli", "run"]
    fields = {
        "dataset": config.dataset, "scenario": config.scenario, "backend": config.backend,
        "backend-config": json.dumps(config.backend_config), "seed": config.seed,
        "split-manifest": config.split_manifest, "sample-manifest": config.sample_manifest,
        "attribute-vectors-root": config.attribute_vectors_root,
        "samples-per-activity": config.samples_per_activity,
        "prototype-metric": config.prototype_metric, "attribute-weighting": config.attribute_weighting,
        "run-root": config.run_root, "run-name": config.run_name, "gpu-id": args.gpu_id,
    }
    for key, value in fields.items():
        result.extend(["--" + key, str(value)])
    if config.reuse_models:
        result.append("--reuse-models")
    if config.evaluate_only:
        result.append("--evaluate-only")
    return result


def validate_dependencies(args):
    required = {MODULES[name] for name in args.backends if name in MODULES} | {"sklearn", "pandas"}
    if "neuralnet_fastai" in args.backends:
        required.update({"fastai.learner", "fastai.layers"})
    versions = {}
    for name in sorted(required):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "import OK")
    if any(name in MODULES for name in args.backends):
        # CUDA visibility is set before importing torch or running a CUDA query.
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"Expected one visible CUDA GPU, physical GPU {args.gpu_id}")
        print(f"CUDA logical 0: {torch.cuda.get_device_name(0)}", flush=True)
    print("Dependencies: " + json.dumps(versions), flush=True)


def snapshot_inputs(args):
    spec, splits, split_path, sample_path = v4.read_protocol(args.dataset)
    root = PROJECT / "configs/attrIbutes_v4" / args.dataset
    vectors, errors = v4.scan_vectors(root, spec=spec, splits=splits)
    if errors:
        raise ValueError("Missing/invalid v4 vectors: " + json.dumps(errors, ensure_ascii=False))
    v4.verify_test_windows(spec, splits, sample_path)
    snapshot = {
        "dataset": args.dataset, "seed": 42, "vectors": vectors,
        "input_sha256": {str(p.resolve()): v4.digest(p) for p in (
            split_path, sample_path, spec.features_path, spec.feature_names_path,
            spec.labels_path, spec.attributes_path, PROJECT / "configs/datasets" / f"{args.dataset}.json",
        )},
    }
    print(f"Validated {len(vectors)} v4 A2 matrices; attribute counts: "
          f"{sorted({len(value['attributes']) for value in vectors.values()})}", flush=True)
    return splits, snapshot


def complete_rows(run_dir, backend, indices, snapshot):
    """A partial/stale/duplicated summary must never cause a backend to be skipped."""
    path = run_dir / "summary.csv"
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(indices) or {int(row["split_index"]) for row in rows} != set(indices):
            return None
        count = 4 * v4.EVAL_COUNTS[snapshot["dataset"]]
        for row in rows:
            index = int(row["split_index"])
            split_dir = run_dir / f"split_{index:03d}"
            manifest = split_dir / "models/attribute_backend_manifest.json"
            source = snapshot["vectors"][index]
            if (row["backend"] != backend or row["scenario"] != "zsl"
                    or row["dataset"] != snapshot["dataset"] or int(row["eval_samples"]) != count
                    or row["attribute_vectors_sha256"] != source["sha256"]
                    or not manifest.is_file() or not (split_dir / "metrics.json").is_file()
                    or not (split_dir / "details.csv").is_file()):
                return None
            if v4.read_json(manifest).get("classifier_backend") != backend:
                return None
            metrics = v4.read_json(split_dir / "metrics.json")
            per_attribute = metrics.get("per_attribute", {})
            if (metrics.get("sample_count") != count
                    or metrics.get("attribute_vectors", {}).get("sha256") != source["sha256"]
                    or set(per_attribute) != set(source["attributes"])):
                return None
            if any(not math.isfinite(float(item["accuracy"]))
                   or not 0 <= float(item["accuracy"]) <= 1 for item in per_attribute.values()):
                return None
            if any(not math.isfinite(float(row[key])) or not 0 <= float(row[key]) <= 1 for key in METRICS):
                return None
        return rows
    except (OSError, KeyError, ValueError, TypeError):
        return None


def write_csv(path, rows, columns):
    target = data_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path(target.with_name(target.name + ".tmp"))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def summarize(args, indices, snapshot, failures):
    from zshar.evaluation.attribute_summary import export_attribute_accuracy

    comparison, all_rows, incomplete = [], [], []
    for backend in args.backends:
        run_dir = data_path(args.output_root / args.dataset / run_name(args.dataset, backend))
        rows = complete_rows(run_dir, backend, indices, snapshot)
        if rows is None:
            incomplete.append(backend)
            continue
        try:
            attributes = export_attribute_accuracy(
                run_dir, expected_split_indices=indices, vector_inputs=snapshot["vectors"],
                expected_sample_count=4 * v4.EVAL_COUNTS[args.dataset], require_complete=True,
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures[backend] = f"Attribute metric validation failed: {exc}"
            incomplete.append(backend)
            continue
        row = {"backend": backend, "split_count": len(rows)}
        for metric in METRICS:
            values = [float(item[metric]) for item in rows]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values)
        row["attribute_mean_accuracy"] = attributes["attribute_mean_accuracy"]
        comparison.append(row)
        all_rows.extend(rows)
    directory = data_path(args.output_root / args.dataset / "comparison")
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["backend", "split_count"] + [name + suffix for name in METRICS for suffix in ("_mean", "_std")]
    write_csv(directory / "comparison.csv", comparison, columns + ["attribute_mean_accuracy"])
    if all_rows:
        write_csv(directory / "all_split_metrics.csv", all_rows, list(all_rows[0]))
    v4.write_json(data_path(directory / "status.json"), {
        "dataset": args.dataset, "seed": 42, "physical_gpu": args.gpu_id,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not incomplete and not failures,
        "completed_backends": [row["backend"] for row in comparison],
        "incomplete_backends": incomplete, "failures": failures,
    })
    print(f"Comparison: {directory}; completed {len(comparison)}/{len(args.backends)} backends", flush=True)
    return incomplete


def parse_backends(value):
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or len(set(result)) != len(result) or set(result) - set(BACKENDS):
        raise argparse.ArgumentTypeError(f"Choose unique backends from {','.join(BACKENDS)}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("pamap2", "mhealth"))
    parser.add_argument("mode", nargs="?", choices=("check", "train-test", "evaluate", "report"), default="train-test")
    parser.add_argument("--gpu-id", type=int, help="Default PAMAP2=0, MHEALTH=1")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backends", type=parse_backends, default=list(BACKENDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.gpu_id is None:
        args.gpu_id = {"pamap2": 0, "mhealth": 1}[args.dataset]
    if args.gpu_id < 0 or args.num_cpus < 1:
        raise ValueError("GPU must be nonnegative and CPU count must be positive")
    prepare_environment(args)
    if args.mode != "report":
        validate_dependencies(args)
    splits, snapshot = snapshot_inputs(args)
    indices = [split.index for split in splits]
    configs = {backend: experiment_config(args, backend) for backend in args.backends}
    print(f"{args.dataset.upper()}: physical GPU {args.gpu_id} -> logical CUDA 0; "
          f"{len(args.backends)} backends x {len(indices)} splits\n"
          f"Output filesystem: {args.output_root} (device {args.output_root.stat().st_dev})", flush=True)
    # Validate every saved protocol before any training or snapshot replacement.
    for backend, config in configs.items():
        run_dir = data_path(args.output_root / args.dataset / config.run_name)
        v4.validate_existing_run(run_dir, config, snapshot)
        if args.mode == "train-test" and not args.resume and any(run_dir.glob("split_*/models")):
            raise ValueError(f"Models already exist: {run_dir}; pass --resume")
    if args.mode == "check":
        print("Preflight passed. No classifier trained or evaluated.", flush=True)
        return 0
    queue_root = data_path(args.output_root / args.dataset)
    with data_path(queue_root / ".benchmark.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another benchmark is using {queue_root}")
        failures = {}
        if args.mode != "report":
            logs = data_path(queue_root / "logs")
            logs.mkdir(parents=True, exist_ok=True)
            for position, (backend, config) in enumerate(configs.items(), 1):
                run_dir = data_path(queue_root / config.run_name)
                print(f"\n[{position}/{len(configs)}] {backend}: {run_dir}", flush=True)
                try:
                    v4.validate_existing_run(run_dir, config, snapshot)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    with data_path(run_dir / ".training.lock").open("a+") as model_lock:
                        fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if args.resume and args.mode == "train-test" and complete_rows(run_dir, backend, indices, snapshot):
                            print("Reusing completed 20-split backend.", flush=True)
                            continue
                        v4.write_json(data_path(run_dir / "v4_input_manifest.json"), snapshot)
                        if args.mode == "train-test":
                            v4.preserve_incomplete_models(run_dir, resume=args.resume)
                        invocation = command(args, config)
                        print("$ " + shlex.join(invocation), flush=True)
                        log_path = data_path(logs / f"{backend}.log")
                        print(f"Backend output: {log_path}", flush=True)
                        with log_path.open("a", encoding="utf-8") as log:
                            log.write(f"\n{datetime.now(timezone.utc).isoformat()} $ {shlex.join(invocation)}\n")
                            log.flush()
                            subprocess.run(invocation, cwd=run_dir, env=os.environ.copy(),
                                           stdout=log, stderr=subprocess.STDOUT, check=True)
                        if complete_rows(run_dir, backend, indices, snapshot) is None:
                            raise RuntimeError("Backend did not produce complete valid 20-split results")
                except Exception as exc:
                    failures[backend] = f"{type(exc).__name__}: {exc}"
                    print(f"FAILED {backend}: {failures[backend]}", flush=True)
                    if args.fail_fast:
                        summarize(args, indices, snapshot, failures)
                        raise
        incomplete = summarize(args, indices, snapshot, failures)
        if incomplete or failures:
            print("Incomplete benchmark; inspect comparison/status.json and logs, then rerun with --resume.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
