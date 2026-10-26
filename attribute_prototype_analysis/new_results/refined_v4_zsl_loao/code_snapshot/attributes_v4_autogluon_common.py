"""Shared v4 independent AutoGluon ZSL launcher for five datasets."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from zshar.workspace_paths import normalize_migrated_paths
EVAL_COUNTS = {"pamap2": 50, "mhealth": 30, "uschad": 30, "wisdm": 30, "dsads": 30}
UNSEEN_COUNTS = {"pamap2": 4, "mhealth": 4, "uschad": 4, "wisdm": 6, "dsads": 6}
DEFAULT_GPUS = {"pamap2": 0, "mhealth": 1, "uschad": 0, "wisdm": 1, "dsads": 0}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(normalize(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(normalize(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest_paths(dataset, seed=42):
    root = PROJECT / "split/zero_shot_split"
    return (root / f"{dataset}_unseen{UNSEEN_COUNTS[dataset]}_random20_splits_seed{seed}.json",
            root / f"{dataset}_unseen{UNSEEN_COUNTS[dataset]}_random20_eval{EVAL_COUNTS[dataset]}_samples_seed{seed}.json")


def default_run_name(dataset, time_limit, seed=42):
    return (f"{dataset}_zsl_autogluon_default_ensemble_a2_v4_independent_"
            f"t{time_limit}_random20_eval{EVAL_COUNTS[dataset]}_seed{seed}")


def read_protocol(dataset, seed=42):
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_split_manifest

    spec = load_dataset_spec(dataset)
    split_path, sample_path = manifest_paths(dataset, seed=seed)
    payload, samples = read_json(split_path), read_json(sample_path)
    for name, manifest in (("split", payload), ("sample", samples)):
        if manifest.get("dataset") != dataset or manifest.get("seed") != seed:
            raise ValueError(f"Wrong dataset/seed in {name} manifest")
    splits = load_split_manifest(split_path, spec)
    indices = [s.index for s in splits]
    if len(indices) != 20 or len(set(indices)) != 20 or payload.get("unseen_count") != UNSEEN_COUNTS[dataset]:
        raise ValueError(f"Expected 20 unique seed-{seed} ZSL splits with {UNSEEN_COUNTS[dataset]} unseen classes")
    if samples.get("samples_per_activity") != EVAL_COUNTS[dataset]:
        raise ValueError("Sample manifest has the wrong per-activity test count")
    if set(map(int, samples.get("samples", {}))) != set(indices):
        raise ValueError("Sample and activity manifests must cover the same 20 splits")
    original = {int(row["split_index"]): row for row in payload["splits"]}
    for split in splits:
        if len(split.unseen_activities) != UNSEEN_COUNTS[dataset]:
            raise ValueError(f"Incorrect unseen class count in split {split.index}")
        if (set(original[split.index]["seen_activities"]) != set(split.seen_activities)
                or set(original[split.index]["unseen_activities"]) != set(split.unseen_activities)):
            raise ValueError(f"Activity names/IDs disagree in split {split.index}")
        by_activity = samples["samples"][str(split.index)]
        if set(by_activity) != set(split.unseen_activities):
            raise ValueError(f"Test activities disagree with unseen activities in split {split.index}")
        all_ids = []
        for ids in by_activity.values():
            if len(ids) != EVAL_COUNTS[dataset] or len(set(map(int, ids))) != len(ids):
                raise ValueError(f"Invalid fixed test samples in split {split.index}")
            all_ids.extend(map(int, ids))
        if len(set(all_ids)) != len(all_ids):
            raise ValueError(f"Duplicate test window across activities in split {split.index}")
    return spec, splits, split_path, sample_path


def scan_vectors(root, *, spec, splits):
    """Only accept complete matrices with validated full-generation provenance."""
    from zshar.data import load_activity_vectors
    from zshar.semantics import completion

    vectors, errors = {}, {}
    for split in splits:
        directory = root / f"split_{split.index}"
        path = directory / "activity_attribute_vectors_a2.json"
        plan_path = directory / "a2_attribute_plan.json"
        try:
            plan = read_json(plan_path)
            names = completion.get_attribute_names(plan)
            if plan.get("dataset") != spec.key or plan.get("split_index") != split.index:
                raise ValueError("Plan dataset/split mismatch")
            meta = plan.get("_meta", {})
            if (set(meta.get("seen", [])) != set(split.seen_activities)
                    or set(meta.get("unseen", [])) != set(split.unseen_activities)):
                raise ValueError("Plan seen/unseen classes disagree with the requested split")
            if not 9 <= len(names) <= 15:
                raise ValueError("Expected 9-15 final attributes")
            values, columns = load_activity_vectors(path)
            if columns != names:
                raise ValueError("Vector attribute order does not match the final plan")
            completion.validate_llm_output(values, list(spec.activity_names), names)
            completion.validate_saved_generation(
                vector_path=path, report_path=directory / "activity_attribute_vectors_a2.report.json",
                plan_path=plan_path, raw_path=directory / "activity_attribute_vectors_a2.raw_response.txt",
                a1_path=spec.attributes_path,
            )
            vectors[split.index] = {
                "path": str(path.resolve()), "sha256": digest(path), "attributes": names,
                "plan_sha256": digest(plan_path),
                "semantic_definitions": {attr["name"]: {"definition": attr["definition"], "levels": attr["levels"]}
                                         for attr in plan["attributes"]},
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[split.index] = str(exc)
    return vectors, errors


def wait_for_vectors(root, *, spec, splits, args):
    started = time.monotonic()
    while True:
        vectors, errors = scan_vectors(root, spec=spec, splits=splits)
        if not errors:
            print(f"Validated all {len(splits)} v4 matrices: {root}", flush=True)
            return vectors
        message = f"{spec.key}: {len(vectors)}/{len(splits)} ready; missing/invalid splits={sorted(errors)}"
        if args.mode != "train-test" or not args.wait_for_vectors:
            raise ValueError(message + "\n" + json.dumps(errors, ensure_ascii=False, indent=2))
        if args.wait_timeout and time.monotonic() - started >= args.wait_timeout:
            raise TimeoutError(message + "; wait timeout reached")
        print(message + f"; checking again in {args.poll_seconds:g}s", flush=True)
        time.sleep(args.poll_seconds)


def verify_test_windows(spec, splits, sample_path):
    from zshar.data import load_dataset_table

    table = load_dataset_table(spec)
    lookup = table.frame.set_index("window_index")["activity_name"]
    samples = read_json(sample_path)["samples"]
    for split in splits:
        for activity, ids in samples[str(split.index)].items():
            window_ids = list(map(int, ids))
            if not set(window_ids).issubset(lookup.index) or not lookup.loc[window_ids].eq(activity).all():
                raise ValueError(f"Test window IDs/classes disagree with data: split {split.index}, {activity}")
    print(f"Verified fixed test windows: {UNSEEN_COUNTS[spec.key] * EVAL_COUNTS[spec.key]} samples per split", flush=True)


def experiment_config(dataset, args, indices):
    from zshar.pipeline.runner import ExperimentConfig

    split_path, sample_path = manifest_paths(dataset, seed=args.seed)
    return ExperimentConfig(
        dataset=dataset, scenario="zsl", backend="autogluon",
        backend_config={
            "time_limit": args.time_limit, "presets": "medium_quality",
            "num_gpus": 1, "num_cpus": args.num_cpus,
            "num_bag_folds": 0, "num_stack_levels": 0,
            "fit_weighted_ensemble": True, "model_suite": "default",
            "lightgbm_device": "cuda", "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
            "verbosity": 2,
        },
        seed=args.seed, split_manifest=str(split_path), sample_manifest=str(sample_path),
        attribute_vectors_root=str(args.attribute_vectors_root), split_indices=set(indices),
        samples_per_activity=EVAL_COUNTS[dataset], prototype_metric="ordinal_probability",
        attribute_weighting="validation_accuracy", run_root=str(args.run_root), run_name=args.run_name,
        reuse_models=args.resume,
    )


def validate_existing_run(run_dir, config, snapshot):
    saved = run_dir / "config.json"
    if saved.is_file():
        ignored = {"reuse_models", "evaluate_only", "dry_run"}
        previous, current = normalize_migrated_paths(normalize(read_json(saved))), normalize_migrated_paths(normalize(asdict(config)))
        # The pipeline serializes sets as lists; their iteration order is not a protocol change.
        for payload in (previous, current):
            if payload.get("split_indices") is not None:
                payload["split_indices"] = sorted(payload["split_indices"])
        if {k: v for k, v in previous.items() if k not in ignored} != {
                k: v for k, v in current.items() if k not in ignored}:
            raise ValueError("Saved training protocol differs; choose a new --run-name")
    source = run_dir / "v4_input_manifest.json"
    if source.is_file() and normalize_migrated_paths(read_json(source)) != normalize_migrated_paths(normalize(snapshot)):
        raise ValueError("V4 vectors or test inputs changed; choose a new --run-name")
    if not source.is_file() and any(run_dir.glob("split_*/models")):
        raise ValueError("Existing models lack a v4 input snapshot; choose a new --run-name")


def preserve_incomplete_models(run_dir, *, resume):
    paths = [p for p in run_dir.glob("split_*/models") if not (p / "attribute_backend_manifest.json").is_file()]
    if paths and not resume:
        raise ValueError("Incomplete models found; pass --resume to archive and retrain them")
    for path in paths:
        target = path.with_name(f"models.incomplete_{time.time_ns()}")
        shutil.move(str(path), str(target))
        print(f"Archived incomplete model directory: {target}", flush=True)


def query_gpu_state(gpu_id):
    """Read the physical GPU by UUID; a failed query never means the GPU is idle."""
    gpu = subprocess.run([
        "nvidia-smi", f"--id={gpu_id}", "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True, timeout=30)
    rows = list(csv.reader(gpu.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ValueError("Unexpected nvidia-smi GPU response")
    uuid, raw_memory = [value.strip() for value in rows[0]]
    memory = float(raw_memory)
    if not uuid.startswith("GPU-") or not math.isfinite(memory) or memory < 0:
        raise ValueError("Invalid GPU UUID or memory reading")
    processes = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True, timeout=30)
    pids = set()
    for row in csv.reader(processes.stdout.strip().splitlines()):
        if len(row) != 2:
            raise ValueError("Unexpected nvidia-smi process response")
        if row[0].strip() == uuid:
            pid = int(row[1].strip())
            if pid != os.getpid():
                pids.add(pid)
    return {"gpu_uuid": uuid, "memory_used_mib": memory, "other_compute_pids": sorted(pids)}


@contextmanager
def wait_for_gpu_slot(args, *, dataset, run_dir):
    """Hold a per-GPU queue lock through training and require consecutive idle polls."""
    lock_dir = PROJECT / "runtime/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"gpu{args.gpu_id}_v4_autogluon.lock"
    with path.open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print(f"Waiting for GPU {args.gpu_id} v4 queue lock: {path}", flush=True)
                time.sleep(args.gpu_poll_seconds)
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"dataset": dataset, "pid": os.getpid(), "run_dir": str(run_dir)}))
        lock.flush()
        stable = 0
        while True:
            try:
                state = query_gpu_state(args.gpu_id)
                idle = not state["other_compute_pids"] and state["memory_used_mib"] <= args.gpu_max_used_mib
                stable = stable + 1 if idle else 0
                write_json(run_dir / "gpu_wait_status.json", {
                    "state": "waiting_for_gpu", "dataset": dataset, "gpu_id": args.gpu_id,
                    "pid": os.getpid(), "idle_checks": stable, **state,
                })
                print(f"GPU {args.gpu_id}: memory={state['memory_used_mib']:g} MiB, "
                      f"other compute PIDs={state['other_compute_pids']}, "
                      f"idle checks={stable}/{args.gpu_idle_checks}", flush=True)
                if stable >= args.gpu_idle_checks:
                    write_json(run_dir / "gpu_wait_status.json", {
                        "state": "training", "dataset": dataset, "gpu_id": args.gpu_id,
                        "pid": os.getpid(), **state,
                    })
                    print(f"GPU {args.gpu_id} is idle; starting {dataset.upper()} ZSL training/test.", flush=True)
                    break
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                stable = 0
                print(f"GPU {args.gpu_id} query failed; training will not start: {exc}", flush=True)
            time.sleep(args.gpu_poll_seconds)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def build_parser(dataset, *, seed=42):
    parser = argparse.ArgumentParser(description=f"{dataset.upper()} v4 independent AutoGluon ZSL, all 20 splits")
    parser.set_defaults(seed=seed)
    parser.add_argument("mode", nargs="?", choices=("check", "train-test", "summarize"), default="train-test")
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPUS[dataset])
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-cpus", type=int, default=8)
    vector_root = PROJECT / "configs/attrIbutes_v4" / dataset
    if seed != 42:
        vector_root = vector_root / f"seed{seed}"
    parser.add_argument("--attribute-vectors-root", type=Path, default=vector_root)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runtime/zshar_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-vectors", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--wait-timeout", type=float, default=0.0, help="Seconds; 0 means unlimited")
    parser.add_argument("--wait-for-gpu", action="store_true", help="Wait for no other compute processes on the physical GPU")
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    parser.add_argument("--gpu-max-used-mib", type=float, default=1024.0)
    parser.add_argument("--gpu-idle-checks", type=int, default=2)
    return parser


def main(argv=None, *, dataset, seed=42):
    if dataset not in EVAL_COUNTS:
        raise ValueError(f"Supported datasets: {tuple(EVAL_COUNTS)}")
    args = build_parser(dataset, seed=seed).parse_args(argv)
    if args.gpu_id < 0 or args.time_limit <= 0 or args.num_cpus <= 0:
        raise ValueError("GPU must be nonnegative; CPU count and time limit must be positive")
    if (not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0
            or not math.isfinite(args.wait_timeout) or args.wait_timeout < 0):
        raise ValueError("Invalid vector wait/poll configuration")
    if (not math.isfinite(args.gpu_poll_seconds) or args.gpu_poll_seconds <= 0
            or not math.isfinite(args.gpu_max_used_mib) or args.gpu_max_used_mib < 0
            or args.gpu_idle_checks < 1):
        raise ValueError("Invalid GPU wait configuration")
    # Set device visibility before importing the pipeline or any backend module.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    args.run_root = args.run_root.expanduser().resolve()
    args.attribute_vectors_root = args.attribute_vectors_root.expanduser().resolve()
    args.run_name = args.run_name or default_run_name(dataset, args.time_limit, seed=seed)
    run_dir = args.run_root / dataset / args.run_name
    from zshar.evaluation.attribute_summary import export_attribute_accuracy

    spec, splits, split_path, sample_path = read_protocol(dataset, seed=seed)
    indices = [s.index for s in splits]
    if args.mode == "summarize":
        snapshot = read_json(run_dir / "v4_input_manifest.json")
        summary = export_attribute_accuracy(
            run_dir, expected_split_indices=indices,
            vector_inputs={int(k): v for k, v in snapshot["vectors"].items()},
            expected_sample_count=UNSEEN_COUNTS[dataset] * EVAL_COUNTS[dataset],
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "per_attribute"}, indent=2))
        return 0
    vectors = wait_for_vectors(args.attribute_vectors_root, spec=spec, splits=splits, args=args)
    verify_test_windows(spec, splits, sample_path)
    config = experiment_config(dataset, args, indices)
    snapshot = {"dataset": dataset, "seed": seed, "vectors": vectors,
                "input_sha256": {str(p.resolve()): digest(p) for p in (
                    split_path, sample_path, spec.features_path, spec.feature_names_path, spec.labels_path, spec.attributes_path)}}
    validate_existing_run(run_dir, config, snapshot)
    print(f"{dataset.upper()}: GPU {args.gpu_id} (logical 0), 20 splits, AutoGluon default ensemble\n"
          f"  A2 vectors: {args.attribute_vectors_root}\n  splits: {split_path}\n  samples: {sample_path}\n"
          f"  output: {run_dir}\n  time limit per attribute: {args.time_limit}s", flush=True)
    if args.mode == "check":
        print("Check complete; no training, evaluation or output writes.")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".training.lock").open("w") as lock, ExitStack() as resources:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another training process is using this run directory")
        validate_existing_run(run_dir, config, snapshot)
        if any(run_dir.glob("split_*/models")) and not args.resume:
            raise ValueError("Models already exist; pass --resume or choose a new --run-name")
        write_json(run_dir / "v4_input_manifest.json", snapshot)
        if args.wait_for_gpu:
            resources.enter_context(wait_for_gpu_slot(args, dataset=dataset, run_dir=run_dir))
        preserve_incomplete_models(run_dir, resume=args.resume)
        from zshar.pipeline.runner import run_experiment

        try:
            run_experiment(spec, config)
        except BaseException:
            # Preserve usable partial statistics when training is interrupted/fails.
            try:
                export_attribute_accuracy(run_dir, expected_split_indices=indices, vector_inputs=vectors,
                                          expected_sample_count=UNSEEN_COUNTS[dataset] * EVAL_COUNTS[dataset])
            except Exception as exc:
                print(f"Partial attribute summary unavailable: {exc}", file=sys.stderr)
            raise
        summary = export_attribute_accuracy(
            run_dir, expected_split_indices=indices, vector_inputs=vectors,
            expected_sample_count=UNSEEN_COUNTS[dataset] * EVAL_COUNTS[dataset], require_complete=True,
        )
        print(f"Complete: {run_dir}\n20-split attribute mean accuracy: {summary['attribute_mean_accuracy']:.6f}", flush=True)
    return 0
