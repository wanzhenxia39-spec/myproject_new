"""Plan split-specific soft classifier-chain orders with an OpenAI-compatible LLM.

The planner deliberately receives semantic split information only: activity names,
the seen/unseen partition, and split-specific A2 prototypes.  Sensor samples,
predictions, and held-out test metrics are never included in the request.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence


PROMPT_VERSION = "pamap2-soft-oof-multi-order-v2"
DEFAULT_BASE_URL = "https://www.kamiapi.top/v1"
DEFAULT_MODEL = "gpt-5.5"
LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_vector_file(root: Path, split_index: int) -> Path:
    candidates = (
        root / f"split_{split_index}" / "activity_attribute_vectors_a2.json",
        root / f"split_{split_index:03d}" / "activity_attribute_vectors_a2.json",
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not existing:
        raise FileNotFoundError(
            f"No refined A2 vector file found for split {split_index} under {root}"
        )
    if len(existing) > 1:
        digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in existing}
        if len(digests) != 1:
            raise ValueError(
                f"Conflicting padded/unpadded A2 files for split {split_index}: {existing}"
            )
    return existing[0]


def _attributes_from_vectors(vectors: dict[str, dict[str, str]]) -> list[str]:
    if not vectors:
        raise ValueError("A2 vector JSON is empty")
    activities = list(vectors)
    attributes = list(vectors[activities[0]])
    expected = set(attributes)
    if len(expected) != len(attributes):
        raise ValueError("The first A2 vector contains duplicate attribute names")
    for activity, vector in vectors.items():
        if set(vector) != expected or len(vector) != len(attributes):
            raise ValueError(f"Inconsistent A2 schema for activity {activity!r}")
    return attributes


def _sorted_levels(values: Sequence[str]) -> list[str]:
    return sorted(set(values), key=lambda item: (LEVEL_ORDER.get(item, 99), item))


def build_semantic_diagnostics(
    vectors: dict[str, dict[str, str]],
    seen_activities: Sequence[str],
    unseen_activities: Sequence[str],
    attributes: Sequence[str],
) -> dict[str, Any]:
    """Compute prompt diagnostics without accessing sensor/test examples."""
    missing_activities = sorted(
        (set(seen_activities) | set(unseen_activities)).difference(vectors)
    )
    if missing_activities:
        raise ValueError(f"Activities missing from A2 vectors: {missing_activities}")

    per_attribute: dict[str, Any] = {}
    for attribute in attributes:
        seen_levels = _sorted_levels(
            [vectors[activity][attribute] for activity in seen_activities]
        )
        unseen_levels = _sorted_levels(
            [vectors[activity][attribute] for activity in unseen_activities]
        )
        unsupported = [level for level in unseen_levels if level not in seen_levels]
        per_attribute[attribute] = {
            "seen_levels": seen_levels,
            "unseen_levels": unseen_levels,
            "unsupported_unseen_levels": unsupported,
            "unseen_level_coverage": (
                round((len(unseen_levels) - len(unsupported)) / len(unseen_levels), 6)
                if unseen_levels
                else 1.0
            ),
            "unseen_distinct_level_count": len(unseen_levels),
            "constant_on_seen": len(seen_levels) <= 1,
            "constant_on_unseen": len(unseen_levels) <= 1,
            "unseen_groups": {
                level: [
                    activity
                    for activity in unseen_activities
                    if vectors[activity][attribute] == level
                ]
                for level in unseen_levels
            },
        }

    pairwise_differences = []
    collisions = []
    for left, right in combinations(unseen_activities, 2):
        differing = [
            attribute
            for attribute in attributes
            if vectors[left][attribute] != vectors[right][attribute]
        ]
        row = {"activities": [left, right], "differing_attributes": differing}
        pairwise_differences.append(row)
        if not differing:
            collisions.append([left, right])

    return {
        "per_attribute": per_attribute,
        "unseen_pairwise_differences": pairwise_differences,
        "unseen_prototype_collisions": collisions,
    }


def build_messages(
    *,
    split_index: int,
    seen_activities: Sequence[str],
    unseen_activities: Sequence[str],
    vectors: dict[str, dict[str, str]],
    attributes: Sequence[str],
    diagnostics: dict[str, Any],
    dataset_metadata: dict[str, Any],
    candidate_count: int = 3,
) -> list[dict[str, str]]:
    system = (
        "You are designing the attribute order of a probabilistic classifier chain "
        "for strict zero-shot human activity recognition. Return valid JSON only. "
        "Do not invent measurements or claim measured accuracy."
    )
    prompt_payload = {
        "dataset": "PAMAP2",
        "split_index": split_index,
        "seen_activities": list(seen_activities),
        "unseen_activities": list(unseen_activities),
        "candidate_attributes": list(attributes),
        "split_specific_a2_prototypes": {
            activity: vectors[activity]
            for activity in list(seen_activities) + list(unseen_activities)
        },
        "semantic_diagnostics": diagnostics,
        "dataset_metadata": dataset_metadata,
    }
    user = f"""
Plan {candidate_count} distinct split-specific candidate orderings. Every ordering must
contain every candidate attribute exactly once.

Pipeline semantics:
- During training, attribute k receives original sensor features plus all previous
  attributes' native AutoGluon bagging OOF probability vectors and normalized entropy.
- During inference, it receives the original features plus all previous attributes'
  predict_proba vectors and normalized entropy. No hard attribute labels are propagated.
- Final activity prediction uses ordinal_probability compatibility with the A2 prototypes.

Design principles that must guide the order:
1. The best order is split-dependent: account for this split's seen/unseen composition,
   semantic-value coverage, unseen prototype separability, and conditional dependencies.
2. Use a sequential greedy/beam-search-like design: consider marginal downstream chain
   utility at each position, not merely a global static ranking.
3. The proper optimization target is outer-seen-only pseudo-ZSL activity-level Top-1,
   Top-3 recall, and Macro-F1 (including stability across folds), not attribute accuracy.
   No measured pseudo-ZSL scores are supplied here, so use this as a design principle,
   clearly mark uncertainty, and never fabricate numerical results.
4. Explicitly penalize an attribute when unseen semantic levels are absent from the seen
   classes, or when it is constant/weakly varying on seen classes. Usually place such
   attributes late unless they add indispensable unseen-prototype separation.
5. Put transferable anchor attributes early when their levels are covered by seen classes,
   their probabilities should be reliable, and they separate important unseen pairs.
6. Do not equate standalone attribute predictability with chain value. Early attributes
   must provide complementary soft information useful to several downstream predictors;
   redundant or fragile attributes should be later.
7. Do not use or assume unseen sensor examples, unseen labels, held-out test results, or
   earlier forward/reverse experiment outcomes. Unseen activity names and semantic A2
   prototypes are legitimate side information in strict ZSL.

Input JSON:
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

Make the candidates meaningfully diverse: vary the early anchor sequence and the tradeoff
between semantic coverage, unseen-pair separation, and complementary downstream chain
information. Do not produce simple reversals or near-duplicates.

Return exactly this JSON shape with exactly {candidate_count} candidates (concise
justifications, no markdown fences):
{{
  "split_index": {split_index},
  "candidates": [
    {{
      "candidate_id": "candidate_01",
      "ordered_attributes": ["an exact permutation of candidate_attributes"],
      "rationale": {{
        "strategy_summary": "concise summary",
        "anchor_attributes": ["early transferable anchors"],
        "late_attributes": ["coverage-limited/redundant attributes"],
        "coverage_warnings": ["concise warnings"],
        "uncertainty": "what requires outer-seen-only pseudo-ZSL validation"
      }}
    }}
  ]
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _chat_completions_url(base_url: str) -> str:
    root = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return root if root.endswith("/chat/completions") else f"{root}/chat/completions"


def _ssl_context(no_verify: bool):
    if no_verify:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def call_openai_compatible(
    messages: Sequence[dict[str, str]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout: float,
    ssl_no_verify: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Call an OpenAI-compatible chat-completions endpoint without extra SDKs."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": float(temperature),
        "stream": False,
    }
    request = urllib.request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "zshar-attribute-order-planner/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_ssl_context(ssl_no_verify)
        ) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"LLM API timed out after {timeout} seconds") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc
    except (ConnectionResetError, http.client.RemoteDisconnected, ssl.SSLError) as exc:
        raise RuntimeError(f"LLM API connection interrupted: {exc}") from exc

    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError("LLM API returned no choices")
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip(), {
        "id": response_json.get("id"),
        "model": response_json.get("model"),
        "usage": response_json.get("usage"),
    }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response contains no JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value


def validate_plan(
    value: dict[str, Any], *, split_index: int, attributes: Sequence[str]
) -> dict[str, Any]:
    if int(value.get("split_index", -1)) != int(split_index):
        raise ValueError(
            f"LLM response split_index mismatch: expected {split_index}, "
            f"found {value.get('split_index')!r}"
        )
    raw_order = value.get("ordered_attributes")
    if not isinstance(raw_order, list):
        raise ValueError("ordered_attributes must be a list")
    order = [str(attribute) for attribute in raw_order]
    expected = [str(attribute) for attribute in attributes]
    if len(order) != len(set(order)):
        raise ValueError("ordered_attributes contains duplicates")
    missing = sorted(set(expected).difference(order))
    extra = sorted(set(order).difference(expected))
    if missing or extra or len(order) != len(expected):
        raise ValueError(
            f"ordered_attributes is not the exact A2 permutation: "
            f"missing={missing}, extra={extra}"
        )
    rationale = value.get("rationale")
    if not isinstance(rationale, dict):
        raise ValueError("rationale must be an object")
    return {"split_index": int(split_index), "ordered_attributes": order, "rationale": rationale}


def validate_candidate_set(
    value: dict[str, Any],
    *,
    split_index: int,
    attributes: Sequence[str],
    candidate_count: int,
) -> dict[str, Any]:
    """Validate and normalize a diverse set of exact attribute permutations."""
    if int(value.get("split_index", -1)) != int(split_index):
        raise ValueError(
            f"LLM response split_index mismatch: expected {split_index}, "
            f"found {value.get('split_index')!r}"
        )
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != candidate_count:
        raise ValueError(f"LLM response must contain exactly {candidate_count} candidates")
    normalized = []
    observed_orders: set[tuple[str, ...]] = set()
    for position, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"Candidate {position} must be an object")
        plan = validate_plan(
            {
                "split_index": split_index,
                "ordered_attributes": raw_candidate.get("ordered_attributes"),
                "rationale": raw_candidate.get("rationale"),
            },
            split_index=split_index,
            attributes=attributes,
        )
        order_key = tuple(plan["ordered_attributes"])
        if order_key in observed_orders:
            raise ValueError("LLM candidate set contains duplicate attribute orders")
        observed_orders.add(order_key)
        normalized.append(
            {
                "candidate_id": f"candidate_{position:02d}",
                "ordered_attributes": plan["ordered_attributes"],
                "rationale": plan["rationale"],
            }
        )
    return {"split_index": int(split_index), "candidates": normalized}


def _parse_split_indices(value: str | None) -> set[int] | None:
    return (
        {int(part.strip()) for part in value.split(",") if part.strip()}
        if value
        else None
    )


def _parser() -> argparse.ArgumentParser:
    project = _project_root()
    parser = argparse.ArgumentParser(
        description="Generate split-specific PAMAP2 soft-OOF attribute orders with an LLM"
    )
    parser.add_argument(
        "--split-manifest",
        default=str(
            project
            / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
        ),
    )
    parser.add_argument(
        "--attribute-vectors-root",
        default=(
            "/home/wanzhen/LLMbasedZeroshotHARAgent/pamap2_pipeline_runs/"
            "pamap2_random20_unseen4_eval50_no_rerank_keep_models_seed42"
        ),
    )
    parser.add_argument(
        "--dataset-metadata", default=str(project / "configs/metadata/pamap2.json")
    )
    parser.add_argument(
        "--output",
        default=str(project / "configs/attributes/pamap2_llm_candidate_orders_seed42.json"),
    )
    parser.add_argument("--work-dir")
    parser.add_argument("--split-indices", help="optional comma-separated pilot subset")
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=int(os.getenv("PAMAP2_LLM_ORDER_CANDIDATE_COUNT", "3")),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    )
    parser.add_argument(
        "--request-timeout", type=float, default=float(os.getenv("OPENAI_TIMEOUT", "300"))
    )
    parser.add_argument(
        "--max-retries", type=int, default=int(os.getenv("OPENAI_MAX_RETRIES", "8"))
    )
    parser.add_argument(
        "--retry-sleep", type=float, default=float(os.getenv("OPENAI_RETRY_SLEEP", "10"))
    )
    parser.add_argument(
        "--retry-backoff", type=float, default=float(os.getenv("OPENAI_RETRY_BACKOFF", "2"))
    )
    parser.add_argument(
        "--retry-max-sleep",
        type=float,
        default=float(os.getenv("OPENAI_RETRY_MAX_SLEEP", "120")),
    )
    parser.add_argument("--force", action="store_true", help="discard cached split responses")
    parser.add_argument("--dry-run", action="store_true", help="write prompts without API calls")
    parser.add_argument("--ssl-no-verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.candidate_count < 2:
        raise ValueError("--candidate-count must be at least 2")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    if args.retry_sleep < 0 or args.retry_backoff < 1 or args.retry_max_sleep < 0:
        raise ValueError("Retry delays require sleep >= 0, backoff >= 1, max-sleep >= 0")
    manifest_path = Path(args.split_manifest).expanduser().resolve()
    vector_root = Path(args.attribute_vectors_root).expanduser().resolve()
    metadata_path = Path(args.dataset_metadata).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output_path.parent / f"{output_path.stem}_artifacts"
    )
    prompt_dir, response_dir = work_dir / "prompts", work_dir / "responses"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(manifest_path)
    if str(manifest.get("dataset")) != "pamap2" or int(manifest.get("seed", -1)) != 42:
        raise ValueError("This planner requires the PAMAP2 seed=42 strict-ZSL manifest")
    all_splits = manifest.get("splits") or []
    selected = _parse_split_indices(args.split_indices)
    if selected is not None:
        known = {int(item["split_index"]) for item in all_splits}
        unknown = sorted(selected.difference(known))
        if unknown:
            raise ValueError(f"Unknown requested split indices: {unknown}")
        split_rows = [item for item in all_splits if int(item["split_index"]) in selected]
    else:
        split_rows = list(all_splits)
    if not split_rows:
        raise ValueError("No splits selected")

    dataset_metadata = _read_json(metadata_path)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.dry_run and not api_key:
        raise ValueError("OPENAI_API_KEY is required; export it via read -rsp first")
    endpoint = _chat_completions_url(args.base_url)
    generated_splits: dict[str, Any] = {}
    failed_splits: list[tuple[int, str]] = []

    print(f"Planner model    : {args.model}", flush=True)
    print(f"Planner endpoint : {endpoint}", flush=True)
    print(f"Selected splits  : {','.join(str(x['split_index']) for x in split_rows)}", flush=True)
    print("Test leakage     : semantic split/A2 data only; no sensor samples or metrics", flush=True)

    for position, split in enumerate(split_rows, start=1):
        split_index = int(split["split_index"])
        seen = [str(activity) for activity in split["seen_activities"]]
        unseen = [str(activity) for activity in split["unseen_activities"]]
        vector_path = _resolve_vector_file(vector_root, split_index)
        vectors = _read_json(vector_path)
        attributes = _attributes_from_vectors(vectors)
        diagnostics = build_semantic_diagnostics(vectors, seen, unseen, attributes)
        messages = build_messages(
            split_index=split_index,
            seen_activities=seen,
            unseen_activities=unseen,
            vectors=vectors,
            attributes=attributes,
            diagnostics=diagnostics,
            dataset_metadata=dataset_metadata,
            candidate_count=args.candidate_count,
        )
        prompt_sha256 = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        prompt_path = prompt_dir / f"split_{split_index:03d}.json"
        _write_json_atomic(
            prompt_path,
            {
                "prompt_version": PROMPT_VERSION,
                "split_index": split_index,
                "messages": messages,
                "prompt_sha256": prompt_sha256,
                "vector_path": str(vector_path),
                "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
            },
        )
        if args.dry_run:
            print(
                f"[{position}/{len(split_rows)}] split={split_index}: prompt written (dry-run)",
                flush=True,
            )
            continue

        response_path = response_dir / f"split_{split_index:03d}.json"
        candidate_set = None
        if response_path.is_file() and not args.force:
            cached = _read_json(response_path)
            if (
                cached.get("prompt_version") == PROMPT_VERSION
                and cached.get("prompt_sha256") == prompt_sha256
                and cached.get("requested_model") == args.model
            ):
                try:
                    candidate_set = validate_candidate_set(
                        cached["parsed"],
                        split_index=split_index,
                        attributes=attributes,
                        candidate_count=args.candidate_count,
                    )
                    print(
                        f"[{position}/{len(split_rows)}] split={split_index}: reused cached plan",
                        flush=True,
                    )
                except (KeyError, TypeError, ValueError):
                    candidate_set = None

        if candidate_set is None:
            last_error: Exception | None = None
            for attempt in range(1, max(1, args.max_retries) + 1):
                try:
                    raw, response_metadata = call_openai_compatible(
                        messages,
                        api_key=api_key,
                        base_url=args.base_url,
                        model=args.model,
                        temperature=args.temperature,
                        timeout=args.request_timeout,
                        ssl_no_verify=args.ssl_no_verify,
                    )
                    candidate_set = validate_candidate_set(
                        extract_json_object(raw),
                        split_index=split_index,
                        attributes=attributes,
                        candidate_count=args.candidate_count,
                    )
                    _write_json_atomic(
                        response_path,
                        {
                            "prompt_version": PROMPT_VERSION,
                            "prompt_sha256": prompt_sha256,
                            "split_index": split_index,
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "endpoint": endpoint,
                            "requested_model": args.model,
                            "temperature": args.temperature,
                            "response_metadata": response_metadata,
                            "raw_response": raw,
                            "parsed": candidate_set,
                        },
                    )
                    print(
                        f"[{position}/{len(split_rows)}] split={split_index}: "
                        f"planned on attempt {attempt}",
                        flush=True,
                    )
                    break
                except Exception as exc:  # retry transport, JSON, and schema failures
                    last_error = exc
                    print(
                        f"[{position}/{len(split_rows)}] split={split_index}: "
                        f"attempt {attempt}/{max(1, args.max_retries)} failed: {exc}",
                        flush=True,
                    )
                    if attempt < max(1, args.max_retries) and args.retry_sleep > 0:
                        delay = min(
                            args.retry_max_sleep,
                            args.retry_sleep * (args.retry_backoff ** (attempt - 1)),
                        )
                        print(f"  retrying after {delay:.1f}s", flush=True)
                        time.sleep(delay)
            if candidate_set is None:
                failed_splits.append(
                    (split_index, f"{type(last_error).__name__}: {last_error}")
                )
                print(
                    f"[{position}/{len(split_rows)}] split={split_index}: deferred after "
                    f"{args.max_retries} failed attempts; continuing with later splits",
                    flush=True,
                )
                continue

        candidates = candidate_set["candidates"]
        generated_splits[str(split_index)] = {
            # Compatibility alias: the pseudo-ZSL selector uses `candidates`, while
            # the former single-order runner can still consume the first proposal.
            "llm": candidates[0]["ordered_attributes"],
            "rationale": candidates[0]["rationale"],
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "order": candidate["ordered_attributes"],
                    "rationale": candidate["rationale"],
                }
                for candidate in candidates
            ],
            "semantic_diagnostics": diagnostics,
            "a2_vector_path": str(vector_path),
            "a2_vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        }

    if args.dry_run:
        print(f"Dry-run complete. Prompts: {prompt_dir}", flush=True)
        return

    if failed_splits:
        failed_text = ", ".join(str(index) for index, _ in failed_splits)
        details = "; ".join(f"split {index}: {error}" for index, error in failed_splits)
        raise RuntimeError(
            f"No valid response after retries for splits [{failed_text}]. Later successful "
            f"responses were cached. Rerun the same command to resume. Details: {details}"
        )

    result = {
        "schema_version": 3,
        "dataset": "pamap2",
        "attribute_set": "A2_refined_per_split",
        "seed": 42,
        "selection_scope": (
            "LLM planning from seen/unseen activity names and semantic A2 prototypes only; "
            "no unseen sensor samples, sample labels, predictions, or test metrics"
        ),
        "planner": {
            "prompt_version": PROMPT_VERSION,
            "endpoint": endpoint,
            "requested_model": args.model,
            "temperature": args.temperature,
            "candidate_count": args.candidate_count,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "principles": [
            "split-specific seen/unseen semantics and dependencies",
            "sequential greedy/beam-search-like marginal chain utility",
            "outer-seen-only pseudo-ZSL activity metrics as the validation target",
            "explicit penalty for unsupported or constant attribute levels",
            "transferable and complementary anchor attributes early",
            "chain value is distinct from standalone attribute predictability",
        ],
        "splits": generated_splits,
    }
    _write_json_atomic(output_path, result)
    print(f"Wrote {len(generated_splits)} validated split orders to {output_path}", flush=True)
    print(f"Prompts/responses retained in {work_dir}", flush=True)


if __name__ == "__main__":
    main()
