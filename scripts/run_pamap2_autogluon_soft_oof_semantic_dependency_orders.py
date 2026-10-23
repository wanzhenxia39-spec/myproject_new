#!/usr/bin/env python3
"""Run PAMAP2 AutoGluon soft-OOF chains with semantic-dependency DP orders.

Ordering is a completed, classifier-free preprocessing step.  For each split,
the order file was constructed only from seen-activity A2 semantic prototypes
using D[i,j] = 1 - H(A_j | A_i) / H(A_j), followed by exact subset dynamic
programming.  This launcher does not fit an AutoGluon model to choose an order.
It uses AutoGluon only as the configurable bagged soft-OOF attribute-chain backend, runs
the DP-optimal forward order first, and runs its exact reverse second.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = (
    PROJECT / "scripts/run_pamap2_autogluon_soft_oof_catboost_orders.py"
)
BASE_SPEC = importlib.util.spec_from_file_location(
    "pamap2_autogluon_soft_oof_order_common", BASE_SCRIPT
)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

SEMANTIC_ORDERS = (
    PROJECT
    / "configs/attribute_orders/pamap2/semantic_dependency_dp_seed42.json"
)
RUN_NAME_PREFIX = "pamap2_zsl_autogluon_soft_oof_a2_semantic_dependency_dp"
EXPECTED_SELECTION_SCOPE = (
    "seen activity semantic prototypes only; one equally weighted row per seen "
    "activity; no sensor samples, fitted classifier, unseen prototype, or test metric"
)


def validate_protocol() -> tuple[Any, list[Any], list[int]]:
    """Validate that the order is the saved classifier-free semantic DP result."""
    from zshar.data import load_activity_vectors

    spec, splits, _, indices = base.common.preflight()
    if not SEMANTIC_ORDERS.is_file():
        raise FileNotFoundError(
            "Missing semantic dependency order file: "
            f"{SEMANTIC_ORDERS}. Run run_pamap2_semantic_dependency_chains.py prepare first."
        )
    payload = base.read_json(SEMANTIC_ORDERS)
    if (
        payload.get("dataset") != "pamap2"
        or payload.get("attribute_set") != "old_pipeline_A2_refined_per_split"
        or int(payload.get("seed", -1)) != 42
        or payload.get("selection_scope") != EXPECTED_SELECTION_SCOPE
        or payload.get("method")
        != "directional normalized conditional-entropy reduction + exact subset DP"
        or payload.get("split_manifest_sha256") != base.file_hash(base.common.SPLITS)
    ):
        raise ValueError(
            f"Invalid semantic dependency DP order metadata: {SEMANTIC_ORDERS}"
        )
    entries = payload.get("splits", {})
    if set(map(int, entries)) != set(indices):
        raise ValueError("Semantic DP order split indices differ from the split manifest")
    for split in splits:
        vector_path = base.common.resolve_vector(split.index)
        _, attributes = load_activity_vectors(vector_path)
        entry = entries[str(split.index)]
        forward = list(map(str, entry.get("forward", [])))
        reverse = list(map(str, entry.get("reverse", [])))
        if (
            entry.get("attribute_vectors_sha256") != base.file_hash(vector_path)
            or list(map(str, entry.get("seen_activities", [])))
            != list(split.seen_activities)
            or len(forward) != len(set(forward))
            or set(forward) != set(attributes)
            or reverse != forward[::-1]
        ):
            raise ValueError(f"Invalid semantic dependency order for split {split.index}")
    return spec, splits, indices


def configure_base() -> None:
    # Reuse the soft-OOF execution, resume, and reporting implementation while
    # replacing only the already-computed order source and its provenance.
    base.CATBOOST_ORDERS = SEMANTIC_ORDERS
    base.ORDER_LABEL = "semantic dependency DP"
    base.ORDER_SUMMARY_METADATA = {
        "order_method": (
            "directional normalized conditional entropy + exact subset dynamic programming"
        ),
        "ranking_backend": None,
        "selection_scope": "seen activity A2 semantic prototypes only",
        "autogluon_used_for_order_generation": False,
    }
    base.RUN_NAME_PREFIX = RUN_NAME_PREFIX
    base.validate_protocol = validate_protocol
    base.__doc__ = __doc__


def main() -> int:
    configure_base()
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
