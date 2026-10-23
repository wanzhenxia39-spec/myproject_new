#!/usr/bin/env python3
"""Generate selected WISDM semantic A2 plans through Kami with retries."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zshar.semantics import planner1  # noqa: E402


DEFAULT_ROOT = (
    PROJECT / "runtime/refinement_semantic_v3/wisdm_gpt_5_6_sol_11_15"
)
DEFAULT_GAP = PROJECT / "runtime/gap/wisdm/a1_gap_normalized_ordinal_seed42.csv"
DEFAULT_A1 = (
    PROJECT / "configs/attributes/activity_attribute_vectors_initial_wisdm_v1_5.json"
)
DEFAULT_ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
DEFAULT_BASE_URL = "https://www.kamiapi.top/v1"
DEFAULT_MODEL = "gpt-5.6-sol"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def gap_rows(path: Path) -> tuple[list[int], dict[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = [int(row["split_index"]) for row in rows]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("GAP CSV must contain nonempty unique split indices")
    return indices, {int(row["split_index"]): row for row in rows}


def selected_indices(value: str, available: Sequence[int]) -> list[int]:
    if not value.strip():
        return list(available)
    requested = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("--split-indices did not contain a split index")
    if len(requested) != len(set(requested)):
        raise ValueError("--split-indices contains duplicates")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown split indices: {unknown}")
    wanted = set(requested)
    return [index for index in available if index in wanted]


def json_list(row: Mapping[str, str], field: str) -> list[str]:
    value = json.loads(row[field])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid {field} for split {row.get('split_index')}")
    return value


def valid_plan(
    path: Path,
    split_index: int,
    row: Mapping[str, str],
    a1_names: list[str],
    ontology: list[dict[str, Any]],
    min_attributes: int,
    max_attributes: int,
) -> tuple[bool, str]:
    try:
        plan = read_json(path)
        planner1.validate_semantic_plan(
            plan,
            a1_ontology=ontology,
            a1_names=a1_names,
            seen=json_list(row, "seen_activities"),
            unseen=json_list(row, "unseen_activities"),
            dataset="wisdm",
            split_index=split_index,
            min_attributes=min_attributes,
            max_attributes=max_attributes,
        )
        meta = plan.get("_meta", {})
        if meta.get("validation_decision") != "validated_semantic_a2":
            return False, "missing validated_semantic_a2 metadata"
        return True, f"{len(plan['attributes'])} validated attributes"
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return False, str(exc)


def planner_command(
    args: argparse.Namespace,
    split_index: int,
    output: Path,
    concepts_json: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "zshar.semantics.planner1",
        "--dataset-name",
        "wisdm",
        "--a1-json",
        str(args.a1_json),
        "--dknn-csv",
        str(args.gap_csv),
        "--a1-ontology-json",
        str(args.ontology_json),
        "--split-index",
        str(split_index),
        "--output-json",
        str(output),
        "--min-attributes",
        str(args.min_attributes),
        "--max-attributes",
        str(args.max_attributes),
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
    if concepts_json is not None:
        command.extend(["--concepts-json", str(concepts_json)])
    return command


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
    parser.add_argument(
        "--split-indices",
        default="",
        help="Comma-separated split indices. Empty means every split in the GAP CSV.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gap-csv", type=Path, default=DEFAULT_GAP)
    parser.add_argument("--a1-json", type=Path, default=DEFAULT_A1)
    parser.add_argument("--ontology-json", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-attributes", type=int, default=11)
    parser.add_argument("--max-attributes", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=20.0)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--inter-split-delay", type=float, default=5.0)
    parser.add_argument("--force", action="store_true", help="Regenerate valid existing plans.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for field in ("root", "gap_csv", "a1_json", "ontology_json"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if not 9 <= args.min_attributes <= args.max_attributes <= 15:
        raise ValueError("Require 9 <= min-attributes <= max-attributes <= 15")
    if args.max_attempts < 1 or args.timeout <= 0 or args.retry_delay < 0:
        raise ValueError("Attempts and timeout must be positive; delay must be nonnegative")
    if args.retry_backoff < 1 or args.inter_split_delay < 0:
        raise ValueError("Backoff must be >= 1 and inter-split delay must be nonnegative")

    available, rows = gap_rows(args.gap_csv)
    selected = selected_indices(args.split_indices, available)
    a1 = read_json(args.a1_json)
    if not isinstance(a1, dict) or not a1:
        raise ValueError("A1 JSON must be a nonempty activity-vector object")
    first_vector = next(iter(a1.values()))
    if not isinstance(first_vector, dict) or not first_vector:
        raise ValueError("A1 JSON does not contain attribute vectors")
    a1_names = list(first_vector)
    ontology = planner1.validate_a1_ontology(
        read_json(args.ontology_json), a1_names
    )

    print(f"Provider: Kami ({args.base_url})")
    print(f"Model: {args.model}")
    print(f"Selected splits: {selected}")

    pending: list[int] = []
    for index in selected:
        output = args.root / f"split_{index}" / "a2_attribute_plan.json"
        valid, detail = valid_plan(
            output, index, rows[index], a1_names, ontology,
            args.min_attributes, args.max_attributes,
        )
        if valid and not args.force:
            print(f"split {index}: SKIP ({detail})")
        else:
            if output.exists() and not valid:
                print(f"split {index}: existing result is invalid and will be retried ({detail})")
            pending.append(index)

    if args.dry_run:
        for index in pending:
            output = args.root / f"split_{index}" / "a2_attribute_plan.json"
            print(f"split {index}: would run {shlex.join(planner_command(args, index, output))}")
        print("Dry run complete; no API calls or output writes were performed.")
        return 0
    failures: list[int] = []
    for position, index in enumerate(pending, start=1):
        split_dir = args.root / f"split_{index}"
        split_dir.mkdir(parents=True, exist_ok=True)
        output = split_dir / "a2_attribute_plan.json"
        command = planner_command(args, index, output)
        succeeded = False

        raw_response = split_dir / "a2_attribute_plan.raw_response.txt"
        if raw_response.is_file():
            print(f"[{position}/{len(pending)}] split {index}: replaying existing raw response")
            replay_log = split_dir / "batch_logs" / "refinement.replay_existing_raw.log"
            replay_code, _ = run_command(
                planner_command(args, index, output, concepts_json=raw_response),
                replay_log,
            )
            valid, detail = valid_plan(
                output, index, rows[index], a1_names, ontology,
                args.min_attributes, args.max_attributes,
            )
            if replay_code == 0 and valid:
                print(f"[split {index}] COMPLETE from existing raw response ({detail})")
                continue

        if not os.getenv("KAMI_API_KEY"):
            raise RuntimeError(
                f"KAMI_API_KEY is required because split {index} could not be recovered "
                "from its existing raw response"
            )
        for attempt in range(1, args.max_attempts + 1):
            print(f"[{position}/{len(pending)}] split {index}: attempt {attempt}/{args.max_attempts}")
            log = split_dir / "batch_logs" / f"refinement.attempt_{attempt}.log"
            return_code, command_output = run_command(command, log)
            valid, detail = valid_plan(
                output, index, rows[index], a1_names, ontology,
                args.min_attributes, args.max_attributes,
            )
            if return_code == 0 and valid:
                print(f"[split {index}] COMPLETE ({detail})")
                succeeded = True
                break
            print(f"[split {index}] failed: exit={return_code}; {detail}")
            if attempt < args.max_attempts:
                delay = args.retry_delay * args.retry_backoff ** (attempt - 1)
                if any(marker in command_output.lower() for marker in ("http 429", "http 524")):
                    delay = max(delay, 120.0)
                print(f"[split {index}] retrying in {delay:.0f}s")
                time.sleep(delay)
        if not succeeded:
            failures.append(index)
        if position < len(pending) and args.inter_split_delay:
            time.sleep(args.inter_split_delay)

    if failures:
        print(f"Failed splits after retries: {failures}")
        return 1
    print(f"All {len(selected)} selected splits have valid semantic A2 plans.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
