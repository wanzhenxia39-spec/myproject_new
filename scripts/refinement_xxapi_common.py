#!/usr/bin/env python3
"""Shared predictability-aware refinement and full A2 generation via xxapi.

Default: all 20 splits, gpt-6.1-sol, temperature=0.7, non-streaming chat completions.
Only NEW_API_KEY is accepted. No classifier training is performed by this script.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from zshar.datasets import load_dataset_spec
from zshar.features.groups import load_groups, resolve_feature_groups
from zshar.semantics import completion, planner1
from zshar.semantics.gap_analysis import run_gap_analysis
from zshar.semantics.gap_comparison import compare_refinement_gap
from zshar.splits import load_split_manifest
from zshar.workspace_paths import normalize_migrated_paths, stable_migrated_path_text


OUTPUT_BASE = PROJECT / "configs/attrIbutes_v4"
ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
SUPPORTED_DATASETS = ("pamap2", "mhealth", "uschad", "wisdm", "dsads")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def stage_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


def archive_stage(directory: Path, stage: str) -> None:
    stem = "a2_attribute_plan" if stage == "refine" else "activity_attribute_vectors_a2"
    files = list(directory.glob(stem + ".*"))
    if files:
        target = directory / "attempt_history" / f"{stage}_{time.time_ns()}"
        target.mkdir(parents=True)
        for source in files:
            if source.is_file():
                shutil.copy2(source, target / source.name)


def require_key() -> str:
    key = os.getenv("NEW_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Export NEW_API_KEY before running pending xxapi requests")
    # completion.main uses the shared resolver, whose first generic alias is LLM_API_KEY.
    # Set it inside this process so unrelated provider credentials cannot shadow NEW_API_KEY.
    os.environ["LLM_API_KEY"] = key
    return key


def configure_proxy_routing(base_url: str, mode: str) -> None:
    """Bypass ambient proxies for this API host while retaining TLS verification.

    Transport routing does not change prompts or invalidate completed vectors.
    Both aliases must agree because Requests prefers lowercase no_proxy.
    """
    if mode == "environment":
        return
    if mode != "direct":
        raise ValueError(f"Unknown proxy mode: {mode}")
    hostname = urlsplit(base_url).hostname
    if not hostname:
        raise ValueError("API base URL must include a hostname")
    entries = []
    for name in ("no_proxy", "NO_PROXY"):
        for entry in os.environ.get(name, "").split(","):
            entry = entry.strip()
            if entry and entry not in entries:
                entries.append(entry)
    if hostname not in entries:
        entries.append(hostname)
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = ",".join(entries)


def retry_stage(*, directory, stage, valid, action, recover, args) -> bool:
    if valid():
        print(f"  {stage}: SKIP (validated checkpoint)", flush=True)
        return True
    try:
        with stage_log(directory / "batch_logs" / f"{stage}.recovery.log"):
            if recover() and valid():
                print(f"  {stage}: recovered a validated response", flush=True)
                return True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"  {stage}: response recovery unavailable ({exc})", flush=True)
    # A missing key is a configuration error, not a reason to retry every split.
    require_key()
    for attempt in range(1, args.max_attempts + 1):
        archive_stage(directory, stage)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log = directory / "batch_logs" / f"{stage}.{stamp}.attempt_{attempt}.log"
        error = ""
        with stage_log(log):
            print(f"  {stage}: attempt {attempt}/{args.max_attempts}", flush=True)
            try:
                action()
                if not valid():
                    raise ValueError("Output did not pass local schema/provenance validation")
                print(f"  {stage}: COMPLETE", flush=True)
                return True
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                error = str(exc).replace(os.getenv("NEW_API_KEY", "\0"), "[REDACTED]")
                print(f"  {stage}: failed ({error})", flush=True)
        if attempt < args.max_attempts:
            delay = args.retry_delay * args.retry_backoff ** (attempt - 1)
            if any(code in error.lower() for code in ("http 429", "http 524")):
                delay = max(delay, 120.0)
            print(f"  retrying in {delay:g}s; log: {log}", flush=True)
            time.sleep(delay)
    return False


def valid_plan(path, *, dataset, split, ontology, a1_names, a1_path, gap_path, args) -> bool:
    try:
        plan = read_json(path)
        planner1.validate_semantic_plan(
            plan, a1_ontology=ontology, a1_names=a1_names,
            seen=list(split.seen_activities), unseen=list(split.unseen_activities),
            dataset=dataset, split_index=split.index,
            min_attributes=args.min_attributes, max_attributes=args.max_attributes,
        )
        meta = plan["_meta"]
        expected = {
            "validation_decision": "validated_semantic_a2", "model": args.model,
            "base_url": args.base_url, "stream": args.stream,
            "a1_sha256": digest(a1_path), "a1_ontology_sha256": digest(ONTOLOGY),
            "split_csv_sha256": digest(gap_path),
        }
        return all(meta.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, TypeError, KeyError):
        return False


def valid_vectors(directory, *, a1_path, activities) -> bool:
    try:
        plan_path = directory / "a2_attribute_plan.json"
        completion.validate_llm_output(
            read_json(directory / "activity_attribute_vectors_a2.json"),
            activities, completion.get_attribute_names(read_json(plan_path)),
        )
        completion.validate_saved_generation(
            vector_path=directory / "activity_attribute_vectors_a2.json",
            report_path=directory / "activity_attribute_vectors_a2.report.json",
            plan_path=plan_path, raw_path=directory / "activity_attribute_vectors_a2.raw_response.txt",
            a1_path=a1_path,
        )
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def recover_vectors(directory, *, a1_path, activities) -> bool:
    plan_path = directory / "a2_attribute_plan.json"
    raw = directory / "activity_attribute_vectors_a2.raw_response.txt"
    if not raw.is_file():
        return False
    completion.validate_generation_response(raw, plan_path, a1_path)
    plan = read_json(plan_path)
    vectors = completion.assemble_full_vectors(
        plan=plan, activities=activities,
        llm_output=completion.extract_json_object(raw.read_text(encoding="utf-8")),
    )
    vector_path = directory / "activity_attribute_vectors_a2.json"
    write_json(vector_path, vectors)
    write_json(directory / "activity_attribute_vectors_a2.report.json", completion.build_generation_report(
        vectors, activities, plan=plan, plan_path=plan_path, vector_path=vector_path,
        raw_path=raw, a1_path=a1_path,
    ))
    return True


def build_parser(dataset, *, seed=42):
    parser = argparse.ArgumentParser(description=f"{dataset.upper()}: {__doc__}")
    parser.set_defaults(seed=seed)
    output_root = OUTPUT_BASE / dataset
    if seed != 42:
        output_root = output_root / f"seed{seed}"
    parser.add_argument("--root", type=Path, default=output_root)
    parser.add_argument("--stage", choices=("all", "refine", "vectors", "compare"), default="all")
    parser.add_argument("--split-indices", default="", help=f"Comma-separated seed-{seed} split IDs; default all 20")
    parser.add_argument("--base-url", default="https://xxapi.info/v1")
    parser.add_argument("--proxy-mode", choices=("direct", "environment"), default="direct",
                        help="Direct API connection by default; environment honors existing proxy settings")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Opt in to streaming; default matches non-streaming curl")
    parser.add_argument("--min-attributes", type=int, default=9)
    parser.add_argument("--max-attributes", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=20.0)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--inter-split-delay", type=float, default=5.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Read inputs and print the plan; no requests or output writes")
    return parser


def configuration_matches(previous, current):
    """Launcher-only dataset extensions do not invalidate completed experiments.

    LLM settings, policy version, inputs, and semantic implementation hashes must
    still match exactly. Retain the launcher digest as audit information.
    """
    ignored = {"launcher_sha256"}
    return (normalize_migrated_paths({k: v for k, v in previous.items() if k not in ignored}) ==
            normalize_migrated_paths({k: v for k, v in current.items() if k not in ignored}))


def split_manifest_path(spec, seed=42):
    return PROJECT / f"split/zero_shot_split/{spec.key}_unseen{spec.unseen_count}_random20_splits_seed{seed}.json"


def main(argv=None, *, dataset="pamap2", seed=42) -> int:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Supported datasets: {SUPPORTED_DATASETS}")
    args = build_parser(dataset, seed=seed).parse_args(argv)
    args.root = args.root.expanduser().resolve()
    args.base_url = args.base_url.rstrip("/")
    if not 9 <= args.min_attributes <= args.max_attributes <= 15:
        raise ValueError("Require 9 <= min-attributes <= max-attributes <= 15")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        raise ValueError("Temperature must be finite and between 0 and 2")
    if (args.max_attempts < 1 or args.timeout <= 0 or args.retry_delay < 0
            or args.retry_backoff < 1 or args.inter_split_delay < 0
            or not all(math.isfinite(value) for value in (
                args.timeout, args.retry_delay, args.retry_backoff, args.inter_split_delay))):
        raise ValueError("Invalid timeout/retry/delay configuration")

    spec = load_dataset_spec(dataset)
    splits_path = split_manifest_path(spec, seed=seed)
    a1_path = spec.attributes_path.resolve()
    a1 = completion.validate_llm_output(read_json(a1_path), list(spec.activity_names),
                                        [item["name"] for item in read_json(ONTOLOGY)["attributes"]])
    a1_names = list(next(iter(a1.values())))
    ontology = planner1.validate_a1_ontology(read_json(ONTOLOGY), a1_names)
    splits = load_split_manifest(splits_path, spec)
    if len(splits) != 20 or len({split.index for split in splits}) != 20 or read_json(splits_path)["seed"] != seed:
        raise ValueError(f"Expected the 20 unique {dataset.upper()} seed-{seed} ZSL splits")
    requested = [int(value.strip()) for value in args.split_indices.split(",") if value.strip()]
    if len(requested) != len(set(requested)) or set(requested) - {split.index for split in splits}:
        raise ValueError(f"Split IDs must be unique and present in the seed-{seed} manifest")
    selected = [split for split in splits if not requested or split.index in requested]
    names_array = np.load(spec.feature_names_path, allow_pickle=True)
    if names_array.ndim != 1:
        raise ValueError("Actual feature-name array must be one-dimensional")
    feature_names = list(map(str, names_array))
    groups = load_groups()
    resolve_feature_groups(feature_names, groups, dataset=dataset)
    configuration = {
        "policy": f"{dataset}_predictability_xxapi_full_a2_v1", "dataset": dataset, "scenario": "zsl", "seed": seed,
        "api_format": "openai", "base_url": args.base_url, "model": args.model,
        "temperature": args.temperature, "stream": args.stream,
        "min_attributes": args.min_attributes, "max_attributes": args.max_attributes,
        "distance_metric": "normalized_ordinal", "top_k": 2,
        "vector_generation_mode": completion.VECTOR_GENERATION_MODE,
        "launcher_sha256": digest(Path(__file__).resolve()),
        "input_files": {str(path): digest(path) for path in (
            a1_path, ONTOLOGY, splits_path, spec.feature_names_path)},
        "groups_sha256": hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest(),
        "implementation_sha256": {name: digest(PROJECT / "src/zshar" / name) for name in (
            "semantics/planner1.py", "semantics/completion.py", "semantics/gap.py",
            "semantics/gap_analysis.py", "semantics/gap_comparison.py", "llm/transport.py")},
    }
    config_path = args.root / "experiment_config.json"
    if config_path.is_file() and not configuration_matches(read_json(config_path), configuration):
        raise ValueError("Existing experiment inputs/settings/implementation differ; choose a new --root")
    if args.root.exists() and any(args.root.iterdir()) and not config_path.is_file():
        raise ValueError("Nonempty output directory has no experiment_config.json; choose a new --root")
    print(json.dumps({**{key: configuration[key] for key in (
        "policy", "seed", "model", "base_url", "temperature", "stream", "min_attributes", "max_attributes")},
        "stage": args.stage, "selected_split_indices": [split.index for split in selected],
        "proxy_mode": args.proxy_mode,
        "output_root": str(args.root)}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Would recompute A1 GAP, then run the requested stages in split order.")
        print("Each pending split needs one semantic-planning response and one complete-matrix response.")
        print("Dry run: no API requests, classifier training or output writes.")
        return 0

    configure_proxy_routing(args.base_url, args.proxy_mode)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is already using this experiment root")
        # Recheck after taking the lock in case another process initialized the root.
        if config_path.is_file() and not configuration_matches(read_json(config_path), configuration):
            raise ValueError("Experiment configuration changed while waiting for the lock")
        write_json(config_path, configuration)
        gap_path = args.root / f"gap/a1_gap_normalized_ordinal_seed{seed}.csv"
        run_gap_analysis(dataset=dataset, a1_json=a1_path, split_manifest=splits_path,
                         seed=seed, top_k=2, output=gap_path)
        # A directory relocation must not invalidate a plan's unchanged GAP
        # evidence solely because the CSV's a1_json label became canonical.
        if gap_path.is_file():
            original_gap = gap_path.read_text(encoding="utf-8")
            stable_gap = stable_migrated_path_text(original_gap)
            if stable_gap != original_gap:
                temporary_gap = gap_path.with_suffix(".csv.tmp")
                temporary_gap.write_text(stable_gap, encoding="utf-8")
                temporary_gap.replace(gap_path)
        failures = []
        state = {"stage": args.stage, "selected_split_indices": [s.index for s in selected], "splits": {}}
        for position, split in enumerate(selected, 1):
            directory = args.root / f"split_{split.index}"
            plan_path = directory / "a2_attribute_plan.json"
            plan_check = lambda: valid_plan(
                plan_path, dataset=dataset, split=split, ontology=ontology, a1_names=a1_names,
                a1_path=a1_path, gap_path=gap_path, args=args)
            print(f"[{position}/{len(selected)}] split {split.index}", flush=True)
            kwargs = dict(
                dataset=dataset, a1_path=a1_path, ontology_path=ONTOLOGY, gap_path=gap_path,
                gap_info=planner1.resolve_split_from_gap_csv(gap_path, [], [], split.index),
                output_path=plan_path, feature_names=feature_names,
                feature_source=str(spec.feature_names_path.resolve()), config=groups,
                model=args.model, temperature=args.temperature, min_attributes=args.min_attributes,
                max_attributes=args.max_attributes, api_format="openai", base_url=args.base_url,
                timeout=args.timeout, stream=args.stream,
            )
            raw_plan = directory / "a2_attribute_plan.raw_response.txt"
            ok = plan_check()
            if args.stage in ("all", "refine"):
                ok = retry_stage(
                    directory=directory, stage="refine", valid=plan_check,
                    action=lambda: planner1.run_split(**kwargs, api_key=require_key()),
                    recover=lambda: (raw_plan.is_file() and bool(planner1.run_split(
                        **kwargs, concepts_path=raw_plan))), args=args,
                )
            if ok and args.stage in ("all", "vectors"):
                command_args = [
                    "--dataset_name", dataset, "--a2_plan_json", str(plan_path),
                    "--a1-json", str(a1_path), "--activities_json", str(a1_path),
                    "--output_json", str(directory / "activity_attribute_vectors_a2.json"),
                    "--api-format", "openai", "--base-url", args.base_url, "--model", args.model,
                    "--temperature", str(args.temperature), "--timeout", str(args.timeout),
                ] + (["--stream"] if args.stream else [])
                ok = retry_stage(
                    directory=directory, stage="vectors",
                    valid=lambda: valid_vectors(directory, a1_path=a1_path, activities=list(a1)),
                    action=lambda: completion.main(command_args),
                    recover=lambda: recover_vectors(directory, a1_path=a1_path, activities=list(a1)),
                    args=args,
                )
            if args.stage == "compare":
                ok = ok and valid_vectors(directory, a1_path=a1_path, activities=list(a1))
            if not ok:
                failures.append(split.index)
                print(f"  FAILED: split {split.index}; check plan/vector logs", flush=True)
            state["splits"][str(split.index)] = "complete" if ok else "failed"
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(args.root / "batch_status.json", state)
            if not ok and args.fail_fast:
                break
            if position < len(selected) and args.stage != "compare":
                time.sleep(args.inter_split_delay)
        if failures:
            print(f"Failed splits: {failures}. Rerun the same command to resume.", flush=True)
            return 1
        if args.stage in ("all", "compare"):
            manifest = read_json(splits_path)
            chosen = {s.index for s in selected}
            manifest["splits"] = [item for item in manifest["splits"] if item["split_index"] in chosen]
            manifest.update(num_splits=len(selected), selected_split_indices=[s.index for s in selected],
                            source_split_manifest=str(splits_path))
            selection_path = args.root / f"gap/selected_seed{seed}_splits.json"
            write_json(selection_path, manifest)
            compare_refinement_gap(dataset=dataset, a1_json=a1_path, a2_vectors_root=args.root,
                                   split_manifest=selection_path, seed=seed, top_k=2,
                                   output_dir=args.root / "gap/comparison")
        print(f"Completed stage={args.stage}, {len(selected)} splits. Output: {args.root}", flush=True)
    return 0

