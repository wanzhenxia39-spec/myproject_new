#!/usr/bin/env python3
"""Generate split-specific PAMAP2 A2 vectors through Kami with retries."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zshar.semantics import completion  # noqa: E402


DEFAULT_ROOT = (
    PROJECT
    / "runtime/refinement_semantic_v3/"
    "pamap2_gpt_5_6_sol_11_15"
)
DEFAULT_GAP = PROJECT / "runtime/gap/pamap2/a1_gap_normalized_ordinal_seed42.csv"
DEFAULT_ACTIVITIES = (
    PROJECT / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json"
)
DEFAULT_BASE_URL = "https://www.kamiapi.top/v1"
DEFAULT_MODEL = "gpt-5.6-sol"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def split_indices(gap_path: Path) -> list[int]:
    with gap_path.open("r", encoding="utf-8", newline="") as handle:
        values = [int(row["split_index"]) for row in csv.DictReader(handle)]
    if len(values) != 20 or len(set(values)) != 20:
        raise ValueError("Expected exactly 20 unique PAMAP2 split indices")
    return values


def selected_indices(value: str, available: Sequence[int]) -> list[int]:
    if not value.strip():
        return list(available)
    requested = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--split-indices contains duplicates")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown split indices: {unknown}")
    wanted = set(requested)
    return [index for index in available if index in wanted]


def artifact_paths(root: Path, split_index: int) -> dict[str, Path]:
    split_dir = root / f"split_{split_index}"
    return {
        "dir": split_dir,
        "plan": split_dir / "a2_attribute_plan.json",
        "vectors": split_dir / "activity_attribute_vectors_a2.json",
        "prompt": split_dir / "activity_attribute_vectors_a2.prompt.txt",
        "raw": split_dir / "activity_attribute_vectors_a2.raw_response.txt",
        "report": split_dir / "activity_attribute_vectors_a2.report.json",
    }


def validate_plan(paths: Mapping[str, Path], split_index: int) -> tuple[dict, list[str]]:
    plan = read_json(paths["plan"])
    names = completion.get_attribute_names(plan)
    if plan.get("dataset") != "pamap2" or int(plan.get("split_index", -1)) != split_index:
        raise ValueError(f"split {split_index}: plan dataset/split mismatch")
    meta = plan.get("_meta", {})
    decision = meta.get("validation_decision") or meta.get("binding_decision")
    if decision not in {"validated_semantic_a2", "validated_a2"}:
        raise ValueError(f"split {split_index}: plan is not validated")
    return plan, names


def valid_vectors(
    paths: Mapping[str, Path], activities: list[str], attributes: list[str]
) -> tuple[bool, str]:
    try:
        vectors = completion.validate_llm_output(
            read_json(paths["vectors"]), activities, attributes
        )
        completion.validate_saved_generation(
            vector_path=paths["vectors"], report_path=paths["report"],
            plan_path=paths["plan"], raw_path=paths["raw"],
        )
        report = read_json(paths["report"])
        if (
            int(report.get("num_activities", -1)) != len(activities)
            or int(report.get("num_attributes", -1)) != len(attributes)
        ):
            return False, "validation report dimensions disagree"
        return True, f"{len(vectors)} activities x {len(attributes)} attributes"
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return False, str(exc)


def recover_raw(
    paths: Mapping[str, Path], activities: list[str], attributes: list[str],
    a1_vectors: Mapping[str, Any],
) -> bool:
    try:
        plan = read_json(paths["plan"])
        completion.validate_generation_response(paths["raw"], paths["plan"])
        raw = paths["raw"].read_text(encoding="utf-8")
        vectors = completion.assemble_full_vectors(
            plan=plan,
            activities=activities,
            llm_output=completion.extract_json_object(raw),
        )
        write_json(paths["vectors"], vectors)
        report = completion.build_generation_report(
            vectors, activities, plan=plan, plan_path=paths["plan"],
            vector_path=paths["vectors"], raw_path=paths["raw"],
        )
        write_json(paths["report"], report)
        return True
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return False


def archive_invalid_raw(path: Path, attempt: int) -> None:
    if not path.is_file():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.name}.failed_attempt_{attempt}_{stamp}")
    counter = 1
    while target.exists():
        target = path.with_name(
            f"{path.name}.failed_attempt_{attempt}_{stamp}_{counter}"
        )
        counter += 1
    shutil.move(path, target)


def completion_command(
    args: argparse.Namespace, paths: Mapping[str, Path]
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "zshar.semantics.completion",
        "--dataset_name",
        "pamap2",
        "--a2_plan_json",
        str(paths["plan"]),
        "--a1-json",
        str(args.activities_json),
        "--activities_json",
        str(args.activities_json),
        "--output_json",
        str(paths["vectors"]),
        "--prompt_out",
        str(paths["prompt"]),
        "--raw_response_out",
        str(paths["raw"]),
        "--report_json",
        str(paths["report"]),
        "--api-format",
        "openai",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--temperature",
        "0",
        "--stream",
        "--timeout",
        str(args.timeout),
    ]


def run_command(command: Sequence[str], log_path: Path) -> tuple[int, str]:
    environment = dict(os.environ)
    for key in (
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"
    ):
        environment.pop(key, None)
    environment["PYTHONPATH"] = str(SRC)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT,
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
            lines.append(line)
        return process.wait(), "".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gap-csv", type=Path, default=DEFAULT_GAP)
    parser.add_argument("--activities-json", type=Path, default=DEFAULT_ACTIVITIES)
    parser.add_argument("--split-indices", default="")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=20.0)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--inter-split-delay", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.root = args.root.expanduser().resolve()
    args.gap_csv = args.gap_csv.expanduser().resolve()
    args.activities_json = args.activities_json.expanduser().resolve()
    if args.max_attempts < 1 or args.timeout <= 0 or args.retry_delay < 0:
        raise ValueError("Attempts and timeout must be positive; delay must be nonnegative")
    if args.retry_backoff < 1 or args.inter_split_delay < 0:
        raise ValueError("Backoff must be >= 1 and inter-split delay must be nonnegative")

    available = split_indices(args.gap_csv)
    selected = selected_indices(args.split_indices, available)
    activity_vectors = read_json(args.activities_json)
    activities = list(map(str, activity_vectors))
    if not activities:
        raise ValueError("Activities JSON is empty")

    prepared: dict[int, tuple[dict[str, Path], list[str]]] = {}
    for index in selected:
        paths = artifact_paths(args.root, index)
        _, attributes = validate_plan(paths, index)
        prepared[index] = (paths, attributes)

    print(f"Provider: Kami ({args.base_url})")
    print(f"Model: {args.model}")
    print(f"Selected splits: {selected}")
    print("Valid semantic plans confirmed for every selected split.")

    pending = []
    for index in selected:
        paths, attributes = prepared[index]
        valid, _ = valid_vectors(paths, activities, attributes)
        if not valid:
            pending.append(index)
    if not args.dry_run and pending and not os.getenv("KAMI_API_KEY"):
        raise RuntimeError("KAMI_API_KEY is required for pending vector-generation calls")

    failures: list[int] = []
    for position, index in enumerate(selected, start=1):
        paths, attributes = prepared[index]
        valid, detail = valid_vectors(paths, activities, attributes)
        if valid:
            print(f"[{position}/{len(selected)}] split {index}: SKIP ({detail})")
            continue

        command = completion_command(args, paths)
        if args.dry_run:
            print(f"[{position}/{len(selected)}] split {index}: would run {shlex.join(command)}")
            continue

        if recover_raw(paths, activities, attributes, activity_vectors):
            print(f"[{position}/{len(selected)}] split {index}: recovered validated raw response")
            continue

        archive_invalid_raw(paths["raw"], 0)

        succeeded = False
        for attempt in range(1, args.max_attempts + 1):
            if attempt > 1:
                archive_invalid_raw(paths["raw"], attempt - 1)
            print(f"[{position}/{len(selected)}] split {index}: attempt {attempt}/{args.max_attempts}")
            log = paths["dir"] / "batch_logs" / f"completion.attempt_{attempt}.log"
            return_code, output = run_command(command, log)
            valid, detail = valid_vectors(paths, activities, attributes)
            if return_code == 0 and valid:
                print(f"[split {index}] COMPLETE ({detail})")
                succeeded = True
                break
            print(f"[split {index}] failed: exit={return_code}; {detail}")
            if attempt < args.max_attempts:
                delay = args.retry_delay * args.retry_backoff ** (attempt - 1)
                if any(marker in output.lower() for marker in ("http 429", "http 524")):
                    delay = max(delay, 120.0)
                print(f"[split {index}] retrying in {delay:.0f}s")
                time.sleep(delay)
        if not succeeded:
            failures.append(index)

        if position < len(selected) and args.inter_split_delay:
            time.sleep(args.inter_split_delay)

    if args.dry_run:
        print("Dry run complete; no API calls or output writes were performed.")
        return 0
    if failures:
        print(f"Failed splits after retries: {failures}")
        return 1
    print(f"All {len(selected)} selected splits have valid A2 vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
