#!/usr/bin/env python3
"""Collect canonical PAMAP2 order configurations into one audited directory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT / "configs/attribute_orders/pamap2"
SOURCES = {
    "autogluon_seen_only_seed42.json": Path(
        "/home/wanzhen/myproject/configs/attributes/pamap_atrribute_order.json"
    ),
    "logistic_regression_seen_only_seed42.json": Path(
        "/home/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs/pamap2/"
        "pamap2_a2_logistic_order_preparation_seed42/attribute_orders_logistic.json"
    ),
    "catboost_seen_only_seed42.json": PROJECT
    / "runtime/pamap2_catboost_chain_seen_only_order_seed42/order_preparation/"
    "attribute_orders_catboost.json",
    "llm_seed42.json": Path(
        "/home/wanzhen/myproject/configs/attributes/pamap2_llm_attribute_order_seed42.json"
    ),
    "llm_candidates_seed42.json": Path(
        "/home/wanzhen/myproject/configs/attributes/pamap2_llm_candidate_orders_seed42.json"
    ),
    "semantic_dependency_dp_seed42.json": DESTINATION
    / "semantic_dependency_dp_seed42.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_order_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("dataset") != "pamap2":
        raise ValueError(f"Not a PAMAP2 order configuration: {path}")
    if int(payload.get("seed", -1)) != 42:
        raise ValueError(f"Not a seed-42 order configuration: {path}")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or len(splits) != 20:
        raise ValueError(f"Expected 20 split-specific orders: {path}")
    return payload


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for filename, source in SOURCES.items():
        destination = DESTINATION / filename
        entry: dict[str, Any] = {
            "name": filename.removesuffix(".json"),
            "source": str(source),
            "copy": str(destination),
        }
        if not source.is_file():
            entry["status"] = "pending_source"
            entries.append(entry)
            continue
        payload = validate_order_payload(source)
        source_hash = sha256(source)
        if source.resolve() != destination.resolve():
            if not destination.is_file() or sha256(destination) != source_hash:
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
        if sha256(destination) != source_hash:
            raise AssertionError(f"Order copy hash mismatch: {destination}")
        entry.update(
            {
                "status": "available",
                "sha256": source_hash,
                "attribute_set": payload.get("attribute_set"),
                "selection_scope": payload.get("selection_scope"),
                "ranking_backend": payload.get("ranking_backend"),
                "split_count": len(payload["splits"]),
                "split_keys": sorted(map(int, payload["splits"])),
            }
        )
        entries.append(entry)
        print(f"Collected {source} -> {destination}")
    write_json(
        DESTINATION / "catalog.json",
        {
            "dataset": "pamap2",
            "seed": 42,
            "policy": (
                "Byte-exact copies of canonical order configurations. Per-run "
                "split_XXX/attribute_order.json files are execution records and are not duplicated."
            ),
            "entries": entries,
        },
    )
    print(f"PAMAP2 order catalog: {DESTINATION / 'catalog.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
