#!/usr/bin/env python3
"""Prepare A1 semantic-DP orders and run PAMAP2 AutoGluon soft-OOF chains.

The order for each seed-42 split is computed only from that split's seen
activities and the shared nine-dimensional A1 v1.5 semantic prototypes.  No
sensor sample, unseen prototype, fitted classifier, or test metric is used for
ordering.  Training runs the DP-optimal order first and its exact reverse
second, using the dataset-profile A1 vectors throughout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE_SCRIPT = PROJECT / "scripts/run_pamap2_autogluon_soft_oof_catboost_orders.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "pamap2_a1_autogluon_soft_oof_common", BASE_SCRIPT
)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

A1 = PROJECT / "configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json"
ONTOLOGY = PROJECT / "configs/attributes/initial_attributes.json"
SPLITS = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"
SAMPLES = PROJECT / "split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json"
ORDER_FILE = (
    PROJECT
    / "configs/attribute_orders/pamap2/initial_a1_v1_5_semantic_dependency_dp_seed42.json"
)
RUN_NAME_PREFIX = "pamap2_zsl_autogluon_soft_oof_a1_v1_5_semantic_dependency_dp"
ATTRIBUTE_SET = "initial_A1_v1_5_9d"
SELECTION_SCOPE = (
    "seen activity A1 semantic prototypes only; one equally weighted row per seen "
    "activity; no sensor samples, fitted classifier, unseen prototype, or test metric"
)
METHOD = "directional normalized conditional-entropy reduction + exact subset DP"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight() -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.splits import load_sample_manifest, load_split_manifest, sample_evaluation_rows

    for path in (A1, ONTOLOGY, SPLITS, SAMPLES):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = load_dataset_spec("pamap2")
    if spec.attributes_path.resolve() != A1.resolve():
        raise ValueError(
            f"PAMAP2 dataset profile must point to the new A1 file: {A1}; "
            f"found {spec.attributes_path}"
        )
    vectors, attributes = load_activity_vectors(A1)
    ontology = read_json(ONTOLOGY)
    expected_attributes = [str(item["name"]) for item in ontology["attributes"]]
    if (
        set(vectors) != set(spec.activity_names)
        or attributes != expected_attributes
        or len(attributes) != 9
    ):
        raise ValueError("PAMAP2 A1 activities or nine-dimensional schema are invalid")

    splits = load_split_manifest(SPLITS, spec)
    indices = [split.index for split in splits]
    if len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 seed-42 splits")
    raw_samples = read_json(SAMPLES)
    if (
        raw_samples.get("dataset") != "pamap2"
        or int(raw_samples.get("seed", -1)) != 42
        or int(raw_samples.get("samples_per_activity", -1)) != 50
        or set(map(int, raw_samples.get("samples", {}))) != set(indices)
    ):
        raise ValueError(f"Invalid PAMAP2 seed-42 eval50 sample manifest: {SAMPLES}")
    sample_manifest = load_sample_manifest(SAMPLES)
    table = load_dataset_table(spec)
    for split in splits:
        evaluation = sample_evaluation_rows(
            table.frame,
            split,
            scenario="zsl",
            samples_per_activity=50,
            seed=42,
            manifest_ids=sample_manifest[split.index],
        )
        if len(evaluation) != 200 or not evaluation.activity_name.value_counts().eq(50).all():
            raise ValueError(f"Split {split.index} does not resolve to 4 x 50 samples")
    return spec, splits, indices


def prepare_orders(splits: Sequence[Any]) -> Path:
    from zshar.data import load_activity_vectors
    from zshar.semantics.dependency_order import semantic_dependency_order

    vectors, attributes = load_activity_vectors(A1)
    entries: dict[str, Any] = {}
    for split in splits:
        seen = list(split.seen_activities)
        prototypes = pd.DataFrame(
            [[vectors[activity][attribute] for attribute in attributes] for activity in seen],
            index=seen,
            columns=attributes,
        )
        result = semantic_dependency_order(prototypes)
        result.update(
            {
                "split_index": split.index,
                "seen_activities": seen,
                "unseen_activities": list(split.unseen_activities),
                "seen_prototypes": {
                    activity: dict(vectors[activity]) for activity in seen
                },
                "attribute_vectors": str(A1.resolve()),
                "attribute_vectors_sha256": sha256(A1),
            }
        )
        entries[str(split.index)] = result
        print(
            f"split {split.index}: K=9, J(forward)={result['forward_objective']:.6f}, "
            f"J(reverse)={result['reverse_objective']:.6f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "dataset": "pamap2",
        "attribute_set": ATTRIBUTE_SET,
        "ontology_version": "1.5",
        "seed": 42,
        "selection_scope": SELECTION_SCOPE,
        "method": METHOD,
        "directional_dependency": "D[i,j] = 1 - H(A_j | A_i) / H(A_j)",
        "objective": "J(pi) = sum_{p<q} D[pi_p,pi_q]",
        "optimizer": "subset dynamic programming, O(K^2 * 2^K)",
        "zero_entropy_target_policy": (
            "D[i,j]=0 when H(A_j)=0 because the target has no uncertainty to reduce"
        ),
        "tie_breaking": "lexicographically earliest order in the A1 schema",
        "forward_definition": "global maximizer of J(pi)",
        "reverse_definition": "exact reversal of the optimal forward order",
        "split_manifest": str(SPLITS.resolve()),
        "split_manifest_sha256": sha256(SPLITS),
        "attribute_vectors": str(A1.resolve()),
        "attribute_vectors_sha256": sha256(A1),
        "ontology": str(ONTOLOGY.resolve()),
        "ontology_sha256": sha256(ONTOLOGY),
        "splits": entries,
    }
    base.write_json(ORDER_FILE, payload)
    print(f"Saved A1 semantic dependency orders: {ORDER_FILE}", flush=True)
    return ORDER_FILE


def validate_protocol() -> tuple[Any, list[Any], list[int]]:
    from zshar.data import load_activity_vectors

    spec, splits, indices = preflight()
    if not ORDER_FILE.is_file():
        raise FileNotFoundError(
            f"Missing A1 semantic dependency order file: {ORDER_FILE}. "
            f"Run `{Path(__file__).name} prepare` first."
        )
    payload = read_json(ORDER_FILE)
    vectors, attributes = load_activity_vectors(A1)
    if (
        payload.get("dataset") != "pamap2"
        or payload.get("attribute_set") != ATTRIBUTE_SET
        or int(payload.get("seed", -1)) != 42
        or payload.get("selection_scope") != SELECTION_SCOPE
        or payload.get("method") != METHOD
        or payload.get("split_manifest_sha256") != sha256(SPLITS)
        or payload.get("attribute_vectors_sha256") != sha256(A1)
        or payload.get("ontology_sha256") != sha256(ONTOLOGY)
    ):
        raise ValueError(f"Invalid A1 semantic dependency order metadata: {ORDER_FILE}")
    entries = payload.get("splits", {})
    if set(map(int, entries)) != set(indices):
        raise ValueError("A1 semantic order split indices differ from the manifest")
    for split in splits:
        entry = entries[str(split.index)]
        forward = list(map(str, entry.get("forward", [])))
        reverse = list(map(str, entry.get("reverse", [])))
        if (
            entry.get("attribute_vectors_sha256") != sha256(A1)
            or list(map(str, entry.get("seen_activities", [])))
            != list(split.seen_activities)
            or set(entry.get("seen_prototypes", {})) != set(split.seen_activities)
            or len(forward) != len(set(forward))
            or set(forward) != set(attributes)
            or reverse != forward[::-1]
        ):
            raise ValueError(f"Invalid A1 semantic dependency order for split {split.index}")
    return spec, splits, indices


def configure_base() -> None:
    base.CATBOOST_ORDERS = ORDER_FILE
    base.ATTRIBUTE_VECTORS_ROOT = None
    base.ATTRIBUTE_SET = ATTRIBUTE_SET
    base.ORDER_LABEL = "nine-dimensional A1 semantic dependency DP"
    base.ORDER_SUMMARY_METADATA = {
        "order_method": (
            "directional normalized conditional entropy + exact subset dynamic programming"
        ),
        "ranking_backend": None,
        "selection_scope": "seen activity A1 semantic prototypes only",
        "autogluon_used_for_order_generation": False,
    }
    base.RUN_NAME_PREFIX = RUN_NAME_PREFIX
    base.validate_protocol = validate_protocol
    base.__doc__ = __doc__


def main() -> int:
    configure_base()
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        if len(sys.argv) != 2:
            raise SystemExit("prepare mode accepts no additional arguments")
        _, splits, _ = preflight()
        prepare_orders(splits)
        validate_protocol()
        return 0
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
