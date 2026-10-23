#!/usr/bin/env python3
"""Run single-stage MHEALTH A2 refinement and vector completion for all splits.

The runner is deliberately sequential because the refinement stage can make several
    long LLM requests. It checkpoints every split, reuses validated plans, and retries
    only the unfinished stage after transient API failures.
API keys are read by the existing planner/completion modules from the shell.
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from zshar.semantics import completion  # noqa: E402
from zshar.semantics.planner1 import (  # noqa: E402
    REFINEMENT_STRATEGY, DEFAULT_MIN_ATTRIBUTES, DEFAULT_MAX_ATTRIBUTES, check_output_strategy,
)


DEFAULT_SPLIT_MANIFEST = (
    "split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json"
)
DEFAULT_GAP_CSV = "runtime/gap/mhealth/a1_gap_normalized_ordinal_seed42.csv"
DEFAULT_A1_JSON = (
    "configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json"
)
DEFAULT_ONTOLOGY_JSON = "configs/attributes/initial_attributes.json"
DEFAULT_OUTPUT_ROOT = "runtime/refinement_semantic_v3/mhealth_gpt_5_6_sol"
DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"
DEFAULT_MODEL = "gpt-5.6-sol"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def readable_json(path: Path, predicate: Callable[[Any], bool] | None = None) -> bool:
    try:
        value = read_json(path)
        return predicate(value) if predicate else True
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def atomic_dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_split_indices(path: Path) -> list[int]:
    manifest = read_json(path)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("splits"), list):
        raise ValueError(f"Split manifest must contain a 'splits' list: {path}")
    indices = [int(row["split_index"]) for row in manifest["splits"]]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError(f"Split manifest has empty or duplicate split indices: {path}")
    declared = manifest.get("num_splits")
    if declared is not None and int(declared) != len(indices):
        raise ValueError(
            f"Split manifest declares {declared} splits but contains {len(indices)}: {path}"
        )
    return indices


def load_gap_indices(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        indices = [int(row["split_index"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GAP CSV lacks valid split_index values: {path}") from exc
    if not indices or len(indices) != len(set(indices)):
        raise ValueError(f"GAP CSV has empty or duplicate split indices: {path}")
    return indices


def parse_selected_splits(value: str, available: Sequence[int]) -> list[int]:
    if not value.strip():
        return list(available)
    requested = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--split-indices contains duplicates")
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested split indices are absent from the manifest: {missing}")
    requested_set = set(requested)
    return [index for index in available if index in requested_set]


def plan_paths(split_dir: Path) -> dict[str, Path]:
    stem = split_dir / "a2_attribute_plan"
    return {
        "plan": stem.with_suffix(".json"),
        "evidence": Path(str(stem) + ".refinement_evidence.json"),
        "prompt_json": Path(str(stem) + ".prompt.json"),
        "prompt_text": Path(str(stem) + ".prompt.txt"),
        "raw": Path(str(stem) + ".raw_response.txt"),
    }


def vector_paths(split_dir: Path) -> dict[str, Path]:
    stem = split_dir / "activity_attribute_vectors_a2"
    return {
        "vectors": stem.with_suffix(".json"),
        "prompt": Path(str(stem) + ".prompt.txt"),
        "raw": Path(str(stem) + ".raw_response.txt"),
        "report": Path(str(stem) + ".report.json"),
    }


def valid_plan(
    paths: Mapping[str, Path],
    split_index: int,
    input_hashes: Mapping[str, str],
) -> tuple[bool, str]:
    try:
        plan = read_json(paths["plan"])
        names = completion.get_attribute_names(plan)
        meta = plan.get("_meta")
        if not isinstance(meta, Mapping):
            return False, "plan has no _meta provenance"
        if meta.get("refinement_strategy") != REFINEMENT_STRATEGY:
            return False, "plan uses a different refinement strategy"
        if plan.get("dataset") != "mhealth" or int(plan.get("split_index")) != split_index:
            return False, "plan dataset or split_index does not match"
        decision = meta.get("validation_decision") or meta.get("binding_decision")
        expected_count = range(meta["min_attributes"], meta["max_attributes"] + 1)
        if decision not in {"validated_semantic_a2", "validated_a2"}:
            return False, f"unrecognized validation decision={decision!r}"
        count_ok = len(names) in expected_count
        if not count_ok:
            return False, f"attribute count {len(names)} conflicts with {decision}"
        provenance_fields = {
            "a1": "a1_sha256",
            "ontology": "a1_ontology_sha256",
            "gap": "split_csv_sha256",
        }
        for source, field in provenance_fields.items():
            if meta.get(field) != input_hashes[source]:
                return False, f"source hash mismatch for {source}"
        return True, str(decision)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return False, f"invalid or incomplete plan: {exc}"


def valid_vectors(
    paths: Mapping[str, Path], plan_path: Path, activities: Sequence[str], a1_path: Path
) -> tuple[bool, str]:
    try:
        plan = read_json(plan_path)
        names = completion.get_attribute_names(plan)
        vectors = read_json(paths["vectors"])
        cleaned = completion.validate_llm_output(
            vectors,
            activities=list(activities),
            expected_attribute_names=names,
        )
        completion.validate_saved_generation(
            vector_path=paths["vectors"], report_path=paths["report"],
            plan_path=plan_path, raw_path=paths["raw"], a1_path=a1_path,
        )
        return True, f"{len(activities)} activities x {len(names)} attributes"
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return False, f"invalid or incomplete vectors: {exc}"


def recover_vectors_from_raw(
    paths: Mapping[str, Path], plan_path: Path, activities: Sequence[str], a1_path: Path
) -> bool:
    """Recover only a complete matrix from the current generation policy."""
    if not paths["raw"].is_file():
        return False
    try:
        plan = read_json(plan_path)
        names = completion.get_attribute_names(plan)
        completion.validate_generation_response(paths["raw"], plan_path, a1_path)
        parsed = completion.extract_json_object(paths["raw"].read_text(encoding="utf-8"))
        cleaned = completion.assemble_full_vectors(
            plan=plan,
            activities=list(activities),
            llm_output=parsed,
        )
        atomic_dump_json(cleaned, paths["vectors"])
        report = completion.build_generation_report(
            cleaned, list(activities), plan=plan, plan_path=plan_path,
            vector_path=paths["vectors"], raw_path=paths["raw"], a1_path=a1_path,
        )
        atomic_dump_json(report, paths["report"])
        return True
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return False


def archive_file(path: Path, reason: str) -> Path | None:
    if not path.is_file():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}.failed_{reason}_{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.failed_{reason}_{stamp}_{counter}")
        counter += 1
    shutil.move(str(path), str(candidate))
    return candidate


def archive_unvalidated_planner_responses(paths: Mapping[str, Path]) -> list[Path]:
    """Preserve truncated/malformed responses before a retry overwrites them."""
    archived: list[Path] = []
    if paths["raw"].is_file() and not readable_json(paths["plan"]):
        moved = archive_file(paths["raw"], "incomplete")
        if moved:
            archived.append(moved)
    return archived


def stream_command(command: Sequence[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join(command)}", flush=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
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
        return_code = process.wait()
    return return_code, "".join(lines)


def retry_delay(error_text: str, args: argparse.Namespace, attempt: int) -> float:
    lowered = error_text.lower()
    cloudflare_or_rate_limit = any(
        marker in lowered
        for marker in ("http 524", "error 524", "retry_after", "http 429")
    )
    base = args.cloud_retry_delay if cloudflare_or_rate_limit else args.retry_delay
    return float(base) * (float(args.retry_backoff) ** max(0, attempt - 1))


def planner_command(args: argparse.Namespace, split_index: int, paths: Mapping[str, Path]) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "zshar.semantics.planner1",
        "--dataset_name",
        "mhealth",
        "--a1-json",
        str(args.a1_json),
        "--a1-ontology-json",
        str(args.a1_ontology_json),
        "--dknn-csv",
        str(args.gap_csv),
        "--split-index",
        str(split_index),
        "--min-attributes",
        str(args.min_attributes),
        "--max-attributes",
        str(args.max_attributes),
        "--output-json",
        str(paths["plan"]),
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
    ]
    if args.stream:
        command.append("--stream")
    return command


def completion_command(
    args: argparse.Namespace, paths: Mapping[str, Path], vector_files: Mapping[str, Path]
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "zshar.semantics.completion",
        "--dataset_name",
        "mhealth",
        "--a2_plan_json",
        str(paths["plan"]),
        "--a1-json",
        str(args.a1_json),
        "--activities_json",
        str(args.a1_json),
        "--output_json",
        str(vector_files["vectors"]),
        "--prompt_out",
        str(vector_files["prompt"]),
        "--raw_response_out",
        str(vector_files["raw"]),
        "--report_json",
        str(vector_files["report"]),
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
    ]
    if args.stream:
        command.append("--stream")
    return command


def run_with_retry(
    *,
    stage: str,
    split_index: int,
    command_factory: Callable[[], list[str]],
    validator: Callable[[], tuple[bool, str]],
    before_attempt: Callable[[], Iterable[Path]] | None,
    args: argparse.Namespace,
    split_dir: Path,
) -> tuple[bool, int, str]:
    last_output = ""
    for attempt in range(1, args.max_attempts + 1):
        if before_attempt:
            for archived in before_attempt():
                print(f"[split {split_index}] preserved incomplete response: {archived.name}")
        log_path = split_dir / "batch_logs" / f"{stage}.attempt_{attempt}.log"
        return_code, last_output = stream_command(command_factory(), log_path)
        valid, detail = validator()
        if return_code == 0 and valid:
            return True, attempt, detail
        print(
            f"[split {split_index}] {stage} attempt {attempt}/{args.max_attempts} "
            f"failed (exit={return_code}; {detail}).",
            flush=True,
        )
        if attempt < args.max_attempts:
            delay = retry_delay(last_output, args, attempt)
            print(f"[split {split_index}] retrying {stage} in {delay:.0f}s.", flush=True)
            time.sleep(delay)
    return False, args.max_attempts, last_output[-2000:]


def load_or_create_manifest(args: argparse.Namespace, split_indices: Sequence[int]) -> dict[str, Any]:
    manifest_path = args.output_root / "batch_manifest.json"
    if readable_json(manifest_path):
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}
    manifest.update(
        {
            "schema_version": "mhealth_a2_batch_v1",
            "dataset": "mhealth",
            "split_manifest": str(args.split_manifest),
            "gap_csv": str(args.gap_csv),
            "all_manifest_split_indices": list(split_indices),
            "settings": {
                "scenario_for_refinement": "zsl",
                "refinement_strategy": REFINEMENT_STRATEGY,
                "min_attributes": args.min_attributes,
                "max_attributes": args.max_attributes,
                "zsl_gzsl_share_class_splits": True,
                "api_format": "openai",
                "base_url": args.base_url,
                "model": args.model,
                "stream": args.stream,
                "max_attempts": args.max_attempts,
            },
            "updated_at": utc_now(),
        }
    )
    manifest.setdefault("splits", {})
    return manifest


def update_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_dump_json(manifest, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially refine bound MHEALTH A2 attributes and generate complete "
            "activity vectors for every experimental split."
        )
    )
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--gap-csv", default=DEFAULT_GAP_CSV)
    parser.add_argument("--a1-json", default=DEFAULT_A1_JSON)
    parser.add_argument("--a1-ontology-json", default=DEFAULT_ONTOLOGY_JSON)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-attributes", type=int, default=DEFAULT_MIN_ATTRIBUTES)
    parser.add_argument("--max-attributes", type=int, default=DEFAULT_MAX_ATTRIBUTES)
    parser.add_argument(
        "--split-indices",
        default="",
        help="Optional comma-separated subset for testing; default is all manifest splits.",
    )
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--model", default=os.getenv("LLM_MODEL", os.getenv("MODEL_NAME", DEFAULT_MODEL))
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use streaming for OpenAI-compatible requests (default). "
            "Pass --no-stream to match a non-streaming chat/completions curl request."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=20.0,
        help="Initial delay for ordinary connection/truncated-response failures.",
    )
    parser.add_argument(
        "--cloud-retry-delay",
        type=float,
        default=120.0,
        help="Initial delay for provider/Cloudflare 524 or 429 responses.",
    )
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--inter-split-delay", type=float, default=5.0)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed split; default records it and continues.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without calling the API or writing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not DEFAULT_MIN_ATTRIBUTES <= args.min_attributes <= args.max_attributes <= DEFAULT_MAX_ATTRIBUTES:
        raise ValueError("Require 9 <= min-attributes <= max-attributes <= 15")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if min(args.retry_delay, args.cloud_retry_delay, args.inter_split_delay) < 0:
        raise ValueError("Retry and inter-split delays cannot be negative")
    if args.retry_backoff < 1:
        raise ValueError("--retry-backoff must be at least 1")

    for field in (
        "split_manifest",
        "gap_csv",
        "a1_json",
        "a1_ontology_json",
        "output_root",
    ):
        setattr(args, field, resolve_path(getattr(args, field)))
    for required in (
        args.split_manifest,
        args.gap_csv,
        args.a1_json,
        args.a1_ontology_json,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest_indices = load_split_indices(args.split_manifest)
    gap_indices = load_gap_indices(args.gap_csv)
    if set(manifest_indices) != set(gap_indices):
        raise ValueError(
            "Split manifest and GAP CSV must contain the same split indices. "
            f"Only in manifest={sorted(set(manifest_indices) - set(gap_indices))}; "
            f"only in GAP={sorted(set(gap_indices) - set(manifest_indices))}"
        )
    selected = parse_selected_splits(args.split_indices, manifest_indices)
    for split_index in selected:
        paths = plan_paths(args.output_root / f"split_{split_index}")
        check_output_strategy(paths["plan"])
        if paths["plan"].is_file():
            meta = read_json(paths["plan"]).get("_meta", {})
            if (meta.get("min_attributes"), meta.get("max_attributes")) != (
                    args.min_attributes, args.max_attributes):
                raise ValueError("Attribute bounds changed; use a new --output-root for this experiment")

    a1_vectors = read_json(args.a1_json)
    if not isinstance(a1_vectors, Mapping) or not a1_vectors:
        raise ValueError(f"A1 JSON must map activity names to vectors: {args.a1_json}")
    activities = list(map(str, a1_vectors.keys()))
    input_hashes = {
        "a1": sha256(args.a1_json),
        "ontology": sha256(args.a1_ontology_json),
        "gap": sha256(args.gap_csv),
    }

    print(f"MHEALTH manifest splits ({len(manifest_indices)}): {manifest_indices}")
    print(f"Selected splits ({len(selected)}): {selected}")
    print(f"Output root: {args.output_root}")
    print("Execution: sequential; completed artifacts are validated and skipped.")

    requires_api = False
    if not args.dry_run:
        for split_index in selected:
            split_dir = args.output_root / f"split_{split_index}"
            check_paths = plan_paths(split_dir)
            plan_ok, decision = valid_plan(check_paths, split_index, input_hashes)
            if not plan_ok:
                requires_api = True
                break
            if decision in {"validated_semantic_a2", "validated_a2"}:
                vectors_ok, _ = valid_vectors(
                    vector_paths(split_dir), check_paths["plan"], activities, args.a1_json
                )
                if not vectors_ok:
                    requires_api = True
                    break
    if requires_api and not completion.resolve_api_key("openai", args.base_url):
        raise RuntimeError(
            "No API key found for the configured OpenAI-compatible endpoint. "
            "For api.zhizengzeng.com, export ZZZ_API_KEY in the current shell."
        )

    batch_manifest_path = args.output_root / "batch_manifest.json"
    batch_manifest = load_or_create_manifest(args, manifest_indices)
    failures: list[int] = []
    work_was_run = False

    for position, split_index in enumerate(selected, start=1):
        split_dir = args.output_root / f"split_{split_index}"
        paths = plan_paths(split_dir)
        vectors = vector_paths(split_dir)
        plan_ok, plan_detail = valid_plan(paths, split_index, input_hashes)
        vectors_ok, vector_detail = (
            valid_vectors(vectors, paths["plan"], activities, args.a1_json)
            if plan_ok
            else (False, "waiting for a valid plan")
        )

        print(
            f"\n[{position}/{len(selected)}] split {split_index}: "
            f"plan={'complete' if plan_ok else 'pending'}, "
            f"vectors={'complete' if vectors_ok else 'pending'}"
        )

        if args.dry_run:
            if not plan_ok:
                print(f"  plan reason: {plan_detail}")
                print("  would run: " + shlex.join(planner_command(args, split_index, paths)))
            if not vectors_ok:
                print("  would run after plan: " + shlex.join(completion_command(args, paths, vectors)))
            continue

        split_dir.mkdir(parents=True, exist_ok=True)
        split_state = batch_manifest["splits"].setdefault(str(split_index), {})
        split_state.update({"started_at": utc_now(), "status": "running"})
        update_manifest(batch_manifest_path, batch_manifest)

        planner_attempts = 0
        if not plan_ok:
            work_was_run = True
            plan_ok, planner_attempts, plan_detail = run_with_retry(
                stage="planner",
                split_index=split_index,
                command_factory=lambda i=split_index, p=paths: planner_command(args, i, p),
                validator=lambda p=paths, i=split_index: valid_plan(p, i, input_hashes),
                before_attempt=lambda p=paths: archive_unvalidated_planner_responses(p),
                args=args,
                split_dir=split_dir,
            )
        else:
            print(f"[split {split_index}] reuse validated plan ({plan_detail}).")

        if plan_ok and not vectors_ok:
            if recover_vectors_from_raw(vectors, paths["plan"], activities, args.a1_json):
                print(f"[split {split_index}] recovered vectors from an existing raw response.")
                vectors_ok, vector_detail = valid_vectors(
                    vectors, paths["plan"], activities, args.a1_json
                )

        completion_attempts = 0
        if plan_ok and not vectors_ok:
            work_was_run = True

            def preserve_bad_vector_raw(v=vector_paths(split_dir)) -> Iterable[Path]:
                if v["raw"].is_file() and not recover_vectors_from_raw(
                    v, paths["plan"], activities, args.a1_json
                ):
                    moved = archive_file(v["raw"], "incomplete")
                    return [moved] if moved else []
                return []

            vectors_ok, completion_attempts, vector_detail = run_with_retry(
                stage="completion",
                split_index=split_index,
                command_factory=lambda p=paths, v=vectors: completion_command(args, p, v),
                validator=lambda p=paths, v=vectors: valid_vectors(
                    v, p["plan"], activities, args.a1_json
                ),
                before_attempt=preserve_bad_vector_raw,
                args=args,
                split_dir=split_dir,
            )
        elif vectors_ok:
            print(f"[split {split_index}] reuse validated vectors ({vector_detail}).")

        split_ok = plan_ok and vectors_ok
        split_state.update(
            {
                "finished_at": utc_now(),
                "status": "complete" if split_ok else "failed",
                "plan": {
                    "status": "complete" if plan_ok else "failed",
                    "detail": plan_detail,
                    "attempts_this_run": planner_attempts,
                    "path": str(paths["plan"]),
                },
                "vectors": {
                    "status": "complete" if vectors_ok else "failed",
                    "detail": vector_detail,
                    "attempts_this_run": completion_attempts,
                    "path": str(vectors["vectors"]),
                },
            }
        )
        update_manifest(batch_manifest_path, batch_manifest)

        if split_ok:
            print(f"[split {split_index}] COMPLETE")
        else:
            failures.append(split_index)
            print(f"[split {split_index}] FAILED; see {split_dir / 'batch_logs'}")
            if args.fail_fast:
                break

        if work_was_run and position < len(selected) and args.inter_split_delay > 0:
            time.sleep(args.inter_split_delay)

    if args.dry_run:
        print("\nDry run complete: no API calls were made and no files were written.")
        return 0

    if failures:
        print(f"\nBatch finished with failed splits: {failures}")
        print("Rerun the same command to resume from validated checkpoints.")
        return 1
    print(f"\nBatch complete: {len(selected)} selected splits have valid plans and vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
