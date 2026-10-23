#!/usr/bin/env python3
"""Generate and validate the five canonical A1 vector files with one LLM setup."""

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
from typing import Any, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from zshar.semantics import initial_vectors  # noqa: E402
from zshar.semantics.completion import validate_llm_output  # noqa: E402
from zshar.semantics.ontology import load_ontology  # noqa: E402


DEFAULT_DATASETS = ("pamap2", "mhealth", "uschad", "wisdm", "dsads")
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runtime/a1_generation_zzz_gpt_5_6_sol"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_datasets(text: str) -> list[str]:
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = sorted(set(values) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}; choose from {DEFAULT_DATASETS}")
    if not values or len(values) != len(set(values)):
        raise ValueError("--datasets must be a nonempty list without duplicates")
    return values


def artifact_paths(output: Path) -> list[Path]:
    return [
        output,
        output.with_suffix(".prompt.json"),
        output.with_suffix(".prompt.txt"),
        output.with_suffix(".raw_response.txt"),
        output.with_suffix(".report.json"),
    ]


def backup_artifacts(datasets: Sequence[str], backup_root: Path) -> list[str]:
    copied: list[str] = []
    for dataset in datasets:
        output = Path(initial_vectors.load_initial_context(dataset)["output_path"])
        destination = backup_root / dataset
        for source in artifact_paths(output):
            if source.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / source.name
                shutil.copy2(source, target)
                copied.append(str(target))
    return copied


def generation_command(args: argparse.Namespace, dataset: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "zshar.semantics.initial_vectors",
        "--dataset",
        dataset,
        "--ontology",
        str(args.ontology),
        "--output",
        str(output),
        "--api-format",
        "openai",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--temperature",
        "0",
        "--timeout",
        str(args.timeout),
        "--stream" if args.stream else "--no-stream",
    ]
    return command


def _append_no_proxy(environment: dict[str, str], host: str) -> None:
    for name in ("NO_PROXY", "no_proxy"):
        values = [item.strip() for item in environment.get(name, "").split(",") if item.strip()]
        if host not in values:
            values.append(host)
        environment[name] = ",".join(values)


def run_logged(
    command: Sequence[str], log_path: Path, *, direct_host: str = ""
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(SRC_ROOT) if not old_pythonpath else f"{SRC_ROOT}{os.pathsep}{old_pythonpath}"
    )
    if direct_host:
        _append_no_proxy(environment, direct_host)
    with log_path.open("w", encoding="utf-8") as log:
        rendered = "$ " + shlex.join(command)
        print(rendered, flush=True)
        log.write(rendered + "\n")
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def validate_generated(
    dataset: str,
    output: Path,
    *,
    model: str,
    base_url: str,
    ontology_path: Path,
) -> tuple[bool, str]:
    report_path = output.with_suffix(".report.json")
    required = artifact_paths(output)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        vectors = json.loads(output.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        context = initial_vectors.load_initial_context(dataset)
        ontology = load_ontology(ontology_path)
        names = [str(attr["name"]) for attr in ontology["attributes"]]
        validate_llm_output(vectors, context["activities"], names)
    except Exception as exc:  # validation should be reported, then retried
        return False, f"invalid output: {exc}"

    expected = {
        "dataset": dataset,
        "attribute_set": "A1",
        "ontology_version": ontology["version"],
        "prompt_policy": initial_vectors.PROMPT_POLICY,
        "ontology_sha256": sha256(ontology_path),
        "prompt_context_sha256": sha256(initial_vectors.DEFAULT_PROMPT_CONTEXT),
        "source": "llm",
        "api_format": "openai",
        "base_url": base_url,
        "model": model,
    }
    mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        return False, f"report provenance mismatch: {mismatches}"
    if report.get("duplicate_full_vectors"):
        return False, f"duplicate full A1 vectors: {report['duplicate_full_vectors']}"
    return True, f"{len(vectors)} activities x {len(names)} attributes"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", initial_vectors.DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", os.getenv("MODEL_NAME", initial_vectors.DEFAULT_MODEL)),
    )
    parser.add_argument("--ontology", type=Path, default=initial_vectors.DEFAULT_ONTOLOGY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help=(
            "Honor HTTP(S)_PROXY for the API request. By default the configured API "
            "host is added to NO_PROXY to avoid local-proxy TLS EOF failures."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all generation commands without API calls or file changes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = parse_datasets(args.datasets)
    args.ontology = args.ontology.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    if not args.ontology.is_file():
        raise FileNotFoundError(args.ontology)
    if args.max_attempts < 1 or args.retry_delay < 0 or args.retry_backoff < 1:
        raise ValueError("Require max-attempts >= 1, retry-delay >= 0, retry-backoff >= 1")

    outputs = {
        dataset: Path(initial_vectors.load_initial_context(dataset)["output_path"]).resolve()
        for dataset in datasets
    }
    print(f"Datasets: {datasets}")
    print(f"Endpoint: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Prompt policy: {initial_vectors.PROMPT_POLICY}")
    api_host = urlsplit(args.base_url).hostname or ""
    print(f"Network: {'environment proxy' if args.use_proxy else f'direct (NO_PROXY={api_host})'}")
    for dataset, output in outputs.items():
        print(f"  {dataset}: {output}")

    if args.dry_run:
        for dataset, output in outputs.items():
            print("would run: " + shlex.join(generation_command(args, dataset, output)))
        print("Dry run complete: no API calls or file changes were made.")
        return 0

    if not os.getenv("ZZZ_API_KEY"):
        raise RuntimeError(
            "ZZZ_API_KEY is not set in this shell. Configure the Zhizengzeng key first."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = args.run_root / "backups" / timestamp
    copied = backup_artifacts(datasets, backup_root)
    print(f"Backed up {len(copied)} existing artifacts to {backup_root}")

    manifest_path = args.run_root / "batch_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "a1_batch_v1",
        "started_at": utc_now(),
        "settings": {
            "datasets": datasets,
            "api_format": "openai",
            "base_url": args.base_url,
            "model": args.model,
            "stream": args.stream,
            "timeout": args.timeout,
            "use_proxy": args.use_proxy,
            "prompt_policy": initial_vectors.PROMPT_POLICY,
            "ontology": str(args.ontology),
            "ontology_sha256": sha256(args.ontology),
            "prompt_context_sha256": sha256(initial_vectors.DEFAULT_PROMPT_CONTEXT),
            "backup_root": str(backup_root),
        },
        "datasets": {},
    }
    atomic_json(manifest, manifest_path)

    failures: list[str] = []
    for position, dataset in enumerate(datasets, start=1):
        output = outputs[dataset]
        print(f"\n[{position}/{len(datasets)}] {dataset}")
        state = manifest["datasets"].setdefault(dataset, {})
        state.update({"status": "running", "started_at": utc_now(), "output": str(output)})
        atomic_json(manifest, manifest_path)
        success = False
        detail = "not attempted"
        attempts = 0
        for attempts in range(1, args.max_attempts + 1):
            log_path = args.run_root / "logs" / f"{dataset}.attempt_{attempts}.log"
            return_code = run_logged(
                generation_command(args, dataset, output),
                log_path,
                direct_host="" if args.use_proxy else api_host,
            )
            if return_code == 0:
                valid, detail = validate_generated(
                    dataset,
                    output,
                    model=args.model,
                    base_url=args.base_url,
                    ontology_path=args.ontology,
                )
            else:
                valid = False
                detail = f"generator exited with code {return_code}; see {log_path}"
            if return_code == 0 and valid:
                success = True
                break
            print(
                f"[{dataset}] attempt {attempts}/{args.max_attempts} failed "
                f"(exit={return_code}; {detail}).",
                flush=True,
            )
            if attempts < args.max_attempts:
                delay = args.retry_delay * (args.retry_backoff ** (attempts - 1))
                print(f"[{dataset}] retrying in {delay:.0f}s.", flush=True)
                time.sleep(delay)

        state.update(
            {
                "status": "complete" if success else "failed",
                "finished_at": utc_now(),
                "attempts": attempts,
                "detail": detail,
            }
        )
        atomic_json(manifest, manifest_path)
        if success:
            print(f"[{dataset}] COMPLETE ({detail})")
        else:
            failures.append(dataset)
            print(f"[{dataset}] FAILED; see {args.run_root / 'logs'}")

    manifest["finished_at"] = utc_now()
    manifest["status"] = "complete" if not failures else "failed"
    manifest["failed_datasets"] = failures
    atomic_json(manifest, manifest_path)
    if failures:
        print(f"\nBatch failed for: {failures}")
        print("Existing files were backed up before generation; rerun the same command to retry.")
        return 1
    print(f"\nAll {len(datasets)} datasets completed. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
