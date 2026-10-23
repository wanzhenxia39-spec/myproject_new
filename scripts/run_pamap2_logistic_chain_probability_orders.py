#!/usr/bin/env python3
"""Run LR hard chains with PAMAP2's saved probability-chain orders.

The launcher reuses the exact forward/reverse order file, split-specific A2
vectors, strict-ZSL split manifest, fixed eval50 sample manifest, prototype
metric, and attribute weighting of the completed AutoGluon soft-OOF runs.  It
runs all selected forward splits before reverse and writes a paired comparison.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT / "scripts/run_pamap2_logistic_regression_chain.py"
SPEC = importlib.util.spec_from_file_location(
    "pamap2_logistic_chain_common", BASE_SCRIPT
)
common = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(common)

PROBABILITY_ORDERS = Path(
    "/home/wanzhen/myproject/configs/attributes/pamap_atrribute_order.json"
)
REFERENCE_ROOT = Path(
    "/home/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs/pamap2"
)
REFERENCE_STEM = (
    "pamap2_zsl_autogluon_soft_oof_a2_{variant}_bag5_20splits_2gpu_seed42_gpu{gpu}"
)
RUN_NAME_PREFIX = (
    "pamap2_zsl_logistic_regression_chain_a2_saved_probability_chain"
)
EXPECTED_ORDER_SHA256 = (
    "9ddc72d3f14dfee049946c1dc84c730ae9c2e63990779b9496bac279721cdc7f"
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_config(variant: str, gpu: int) -> tuple[Path, dict[str, Any]]:
    run_dir = REFERENCE_ROOT / REFERENCE_STEM.format(variant=variant, gpu=gpu)
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed probability-chain config: {path}")
    return run_dir, read_json(path)


def validate_probability_chain_protocol() -> list[int]:
    """Validate exact compatibility with the completed probability-chain runs."""
    from zshar.data import load_activity_vectors

    split_payload = read_json(common.SPLITS)
    sample_payload = read_json(common.SAMPLES)
    order_payload = read_json(PROBABILITY_ORDERS)
    indices = [int(row["split_index"]) for row in split_payload.get("splits", [])]
    expected = set(indices)
    if (
        split_payload.get("dataset") != "pamap2"
        or int(split_payload.get("seed", -1)) != 42
        or len(indices) != 20
        or len(expected) != 20
    ):
        raise ValueError(
            f"Expected myproject_new's fixed PAMAP2 20-split seed-42 manifest: "
            f"{common.SPLITS}"
        )
    if (
        sample_payload.get("dataset") != "pamap2"
        or int(sample_payload.get("seed", -1)) != 42
        or int(sample_payload.get("samples_per_activity", -1)) != 50
        or set(map(int, sample_payload.get("samples", {}))) != expected
    ):
        raise ValueError(f"Expected the PAMAP2 seed-42 eval50 manifest: {common.SAMPLES}")
    for index in indices:
        by_activity = sample_payload["samples"][str(index)]
        if len(by_activity) != 4 or any(
            len(values) != 50 or len(set(map(int, values))) != 50
            for values in by_activity.values()
        ):
            raise ValueError(
                f"Split {index} must contain four unseen activities with 50 unique samples"
            )

    if file_hash(PROBABILITY_ORDERS) != EXPECTED_ORDER_SHA256:
        raise ValueError(
            "The saved probability-chain order file changed; refusing an unmatched rerun"
        )
    if (
        order_payload.get("dataset") != "pamap2"
        or int(order_payload.get("seed", -1)) != 42
        or order_payload.get("attribute_set") != "A2_refined_per_split"
        or set(map(int, order_payload.get("splits", {}))) != expected
    ):
        raise ValueError(
            f"Expected the saved PAMAP2 probability-chain A2 orders: {PROBABILITY_ORDERS}"
        )

    reference_vector_hashes: dict[int, str] = {}
    reference_workers: dict[str, set[int]] = {}
    for variant in common.VARIANTS:
        worker_indices: list[set[int]] = []
        for gpu in (0, 1):
            run_dir, config = reference_config(variant, gpu)
            if (
                config.get("dataset") != "pamap2"
                or config.get("scenario") != "zsl"
                or config.get("backend") != "autogluon_soft_oof"
                or int(config.get("seed", -1)) != 42
                or config.get("attribute_order_variant") != variant
                or int(config.get("samples_per_activity", -1)) != 50
                or config.get("prototype_metric") != "ordinal_probability"
                or config.get("attribute_weighting") != "validation_accuracy"
                or Path(config.get("attribute_vectors_root", "")).resolve()
                != common.A2_ROOT.resolve()
                or file_hash(config.get("attribute_order_config", ""))
                != EXPECTED_ORDER_SHA256
                or file_hash(config.get("split_manifest", ""))
                != file_hash(common.SPLITS)
                or file_hash(config.get("sample_manifest", ""))
                != file_hash(common.SAMPLES)
            ):
                raise ValueError(
                    f"Reference probability-chain protocol mismatch: {run_dir / 'config.json'}"
                )
            configured_indices = {int(value) for value in config.get("split_indices", [])}
            if len(configured_indices) != 10:
                raise ValueError(f"Reference worker must contain ten splits: {run_dir}")
            worker_indices.append(configured_indices)

            summary_path = run_dir / "summary.csv"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"Missing completed probability-chain summary: {summary_path}"
                )
            rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
            if (
                len(rows) != 10
                or {int(row["split_index"]) for row in rows} != configured_indices
                or {row["attribute_order_variant"] for row in rows} != {variant}
                or {row["attribute_order_sha256"] for row in rows}
                != {EXPECTED_ORDER_SHA256}
                or {int(row["eval_samples"]) for row in rows} != {200}
            ):
                raise ValueError(f"Incomplete or mismatched reference summary: {summary_path}")
            for row in rows:
                index = int(row["split_index"])
                vector_hash = str(row["attribute_vectors_sha256"])
                previous = reference_vector_hashes.setdefault(index, vector_hash)
                if previous != vector_hash:
                    raise ValueError(
                        f"Reference probability-chain A2 hash differs for split {index}"
                    )
        if worker_indices[0] & worker_indices[1] or worker_indices[0] | worker_indices[1] != expected:
            raise ValueError(
                f"Reference {variant} workers do not cover all 20 splits exactly once"
            )
        reference_workers[variant] = worker_indices[0] | worker_indices[1]

    for index in indices:
        vector_path = common.resolve_vector(index)
        if file_hash(vector_path) != reference_vector_hashes.get(index):
            raise ValueError(
                f"Current A2 vector differs from the completed probability chain: split {index}"
            )
        _, attributes = load_activity_vectors(vector_path)
        row = order_payload["splits"][str(index)]
        forward = list(map(str, row.get("forward", [])))
        reverse = list(map(str, row.get("reverse", [])))
        if (
            len(forward) != len(set(forward))
            or set(forward) != set(attributes)
            or reverse != forward[::-1]
        ):
            raise ValueError(f"Invalid saved probability-chain order for split {index}")
    return indices


def main() -> int:
    # Reuse the tested sequential runner while replacing only the order source,
    # protocol validator, and output namespace.
    common.ORDERS = PROBABILITY_ORDERS
    common.RUN_NAME_PREFIX = RUN_NAME_PREFIX
    common.validate_protocol = validate_probability_chain_protocol
    common.__doc__ = __doc__
    return int(common.main())


if __name__ == "__main__":
    raise SystemExit(main())
