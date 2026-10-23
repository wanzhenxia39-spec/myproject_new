"""Select LLM-proposed soft-chain orders by outer-seen-only pseudo-ZSL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from zshar.data import load_dataset_table, replace_activity_vectors
from zshar.datasets import load_dataset_spec
from zshar.evaluation.zsl import evaluate_zsl
from zshar.models.serialization import MANAGER_MANIFEST
from zshar.pipeline.calibration import select_pseudo_unseen_partitions
from zshar.pipeline.training import load_or_train_attribute_models
from zshar.splits import load_split_manifest, select_splits


METRICS = (
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
)
DEFAULT_WEIGHTS = {
    "activity_top1_accuracy": 0.4,
    "activity_top3_recall": 0.2,
    "activity_macro_f1": 0.4,
}


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _parse_split_indices(value: str | None) -> set[int] | None:
    return (
        {int(part.strip()) for part in value.split(",") if part.strip()}
        if value
        else None
    )


def _resolve_vector_file(root: Path, split_index: int) -> Path:
    for path in (
        root / f"split_{split_index}" / "activity_attribute_vectors_a2.json",
        root / f"split_{split_index:03d}" / "activity_attribute_vectors_a2.json",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing split {split_index} refined A2 vectors under {root}")


def load_split_candidates(
    payload: Mapping[str, Any],
    *,
    split_index: int,
    attributes: Sequence[str],
) -> list[dict[str, Any]]:
    if str(payload.get("dataset")) != "pamap2" or int(payload.get("seed", -1)) != 42:
        raise ValueError("Candidate configuration must be PAMAP2 seed=42")
    split_payload = (payload.get("splits") or {}).get(str(split_index))
    if not isinstance(split_payload, Mapping):
        raise ValueError(f"Candidate configuration has no split {split_index}")
    candidates = split_payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError(
            f"Split {split_index} needs at least two LLM candidates; regenerate with "
            "the multi-candidate planner"
        )
    expected = set(str(item) for item in attributes)
    result = []
    observed: set[tuple[str, ...]] = set()
    observed_ids: set[str] = set()
    for position, raw in enumerate(candidates, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Split {split_index} candidate {position} is not an object")
        order = [str(item) for item in raw.get("order", [])]
        if len(order) != len(attributes) or len(order) != len(set(order)) or set(order) != expected:
            raise ValueError(
                f"Split {split_index} candidate {position} is not an exact A2 permutation"
            )
        key = tuple(order)
        if key in observed:
            raise ValueError(f"Split {split_index} contains duplicate candidate orders")
        observed.add(key)
        candidate_id = str(raw.get("candidate_id") or f"candidate_{position:02d}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id):
            raise ValueError(f"Unsafe candidate_id for split {split_index}: {candidate_id!r}")
        if candidate_id in observed_ids:
            raise ValueError(f"Duplicate candidate_id for split {split_index}: {candidate_id}")
        observed_ids.add(candidate_id)
        result.append(
            {
                "candidate_id": candidate_id,
                "order": order,
                "rationale": dict(raw.get("rationale") or {}),
            }
        )
    return result


def build_pseudo_zsl_fold_frames(
    frame: pd.DataFrame,
    *,
    pseudo_seen: Sequence[str],
    pseudo_unseen: Sequence[str],
    samples_per_activity: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return class-disjoint training and balanced pseudo-unseen evaluation frames."""
    if samples_per_activity <= 0:
        raise ValueError("samples_per_activity must be positive")
    training = frame[frame["activity_name"].isin(pseudo_seen)].copy()
    if set(training["activity_name"].astype(str)) != set(pseudo_seen):
        missing = sorted(set(pseudo_seen).difference(training["activity_name"].astype(str)))
        raise ValueError(f"Pseudo-seen activities have no training rows: {missing}")
    evaluation_parts = []
    for offset, activity in enumerate(pseudo_unseen):
        group = frame[frame["activity_name"] == activity]
        if len(group) < samples_per_activity:
            raise ValueError(
                f"Pseudo-unseen activity {activity!r} has {len(group)} samples; "
                f"need {samples_per_activity}"
            )
        evaluation_parts.append(
            group.sample(n=samples_per_activity, random_state=seed + offset * 1009)
        )
    evaluation = pd.concat(evaluation_parts).sort_values("window_index").copy()
    if set(training["activity_name"]) & set(evaluation["activity_name"]):
        raise AssertionError("Pseudo-seen/pseudo-unseen activity leakage")
    if set(training["window_index"].astype(int)) & set(
        evaluation["window_index"].astype(int)
    ):
        raise AssertionError("Pseudo-ZSL training/evaluation window leakage")
    return training, evaluation


def aggregate_candidate_folds(
    fold_rows: Sequence[Mapping[str, float]],
    *,
    metric_weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    stability_penalty: float = 0.1,
) -> dict[str, Any]:
    if not fold_rows:
        raise ValueError("At least one pseudo-ZSL fold is required")
    means, standard_deviations = {}, {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in fold_rows], dtype=float)
        means[metric] = float(np.mean(values))
        standard_deviations[metric] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    mean_utility = sum(metric_weights[name] * means[name] for name in METRICS)
    instability = sum(
        metric_weights[name] * standard_deviations[name] for name in METRICS
    )
    return {
        "fold_count": len(fold_rows),
        "metric_means": means,
        "metric_standard_deviations": standard_deviations,
        "mean_utility": float(mean_utility),
        "weighted_instability": float(instability),
        "selection_score": float(mean_utility - stability_penalty * instability),
    }


def select_best_candidate(candidate_summaries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidate_summaries:
        raise ValueError("No candidate summaries")
    return max(
        candidate_summaries,
        key=lambda row: (
            float(row["aggregate"]["selection_score"]),
            float(row["aggregate"]["metric_means"]["activity_macro_f1"]),
            float(row["aggregate"]["metric_means"]["activity_top1_accuracy"]),
            float(row["aggregate"]["metric_means"]["activity_top3_recall"]),
            str(row["candidate_id"]),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    project = _project_root()
    parser = argparse.ArgumentParser(
        description="Select PAMAP2 LLM attribute orders with outer-seen-only pseudo-ZSL"
    )
    parser.add_argument("--dataset-config", default=str(project / "configs/datasets/pamap2.json"))
    parser.add_argument(
        "--split-manifest",
        default=str(project / "split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json"),
    )
    parser.add_argument(
        "--attribute-vectors-root",
        default=(
            "/home/wanzhen/LLMbasedZeroshotHARAgent/pamap2_pipeline_runs/"
            "pamap2_random20_unseen4_eval50_no_rerank_keep_models_seed42"
        ),
    )
    parser.add_argument(
        "--candidate-config",
        default=str(project / "configs/attributes/pamap2_llm_candidate_orders_seed42.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--split-indices")
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--pseudo-unseen-count", type=int, default=4)
    parser.add_argument("--samples-per-activity", type=int, default=50)
    parser.add_argument("--stability-penalty", type=float, default=0.1)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--model-suite", choices=("default", "zshar_five"), default="default")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-id")
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--lightgbm-device", choices=("auto", "cpu", "gpu", "cuda"), default="cuda")
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.fold_count < 2:
        raise ValueError("--fold-count must be at least 2")
    if args.num_bag_folds < 2:
        raise ValueError("--num-bag-folds must be at least 2 for soft OOF")
    if args.stability_penalty < 0:
        raise ValueError("--stability-penalty must be non-negative")

    candidate_path = Path(args.candidate_config).expanduser().resolve()
    candidate_payload = _read_json(candidate_path)
    candidate_sha = _sha256(candidate_path)
    vector_root = Path(args.attribute_vectors_root).expanduser().resolve()
    selection_root = Path(args.selection_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    spec = load_dataset_spec("pamap2", args.dataset_config)
    splits = load_split_manifest(args.split_manifest, spec)
    splits = select_splits(splits, _parse_split_indices(args.split_indices))
    if not splits:
        raise ValueError("No splits selected")
    base_table = load_dataset_table(spec)

    backend_base = {
        "time_limit": args.time_limit,
        "presets": args.presets,
        "num_gpus": args.num_gpus,
        "num_cpus": args.num_cpus,
        "num_bag_folds": args.num_bag_folds,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": True,
        "model_suite": args.model_suite,
        "consider_labels_correlation": True,
        "uncertainty_metric": "normalized_entropy",
        "lightgbm_device": args.lightgbm_device,
        "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"],
        "verbosity": args.verbosity,
    }
    protocol = {
        "name": "outer_seen_only_pseudo_zsl_order_selection",
        "fold_count": args.fold_count,
        "pseudo_unseen_classes_per_fold": args.pseudo_unseen_count,
        "evaluation_samples_per_activity": args.samples_per_activity,
        "prototype_metric": "ordinal_probability",
        "attribute_weighting": "validation_accuracy",
        "metric_weights": DEFAULT_WEIGHTS,
        "stability_penalty": args.stability_penalty,
        "selection_formula": (
            "0.4*mean_top1 + 0.2*mean_top3 + 0.4*mean_macro_f1 - "
            "stability_penalty*(0.4*sd_top1 + 0.2*sd_top3 + 0.4*sd_macro_f1)"
        ),
        "backend": "autogluon_soft_oof",
        "backend_config": backend_base,
    }
    selection_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(selection_root / "protocol.json", protocol)
    selected_splits: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []

    print(f"Candidate config : {candidate_path}", flush=True)
    print(f"Selected splits  : {','.join(str(split.index) for split in splits)}", flush=True)
    print(
        f"Protocol         : {args.fold_count} folds, {args.pseudo_unseen_count} "
        f"pseudo-unseen classes/fold, {args.samples_per_activity} samples/activity",
        flush=True,
    )
    print(
        "Leakage guard    : final-unseen sensor rows never enter selection "
        "training/evaluation frames",
        flush=True,
    )

    for split_position, split in enumerate(splits, start=1):
        vector_path = _resolve_vector_file(vector_root, split.index)
        table = replace_activity_vectors(base_table, vector_path)
        candidates = load_split_candidates(
            candidate_payload,
            split_index=split.index,
            attributes=table.attribute_columns,
        )
        outer_seen = table.frame[
            table.frame["activity_name"].isin(split.seen_activities)
        ].copy()
        if set(outer_seen["activity_name"].astype(str)) != set(split.seen_activities):
            raise ValueError(f"Split {split.index} has incomplete outer-seen data")
        partitions = select_pseudo_unseen_partitions(
            split.seen_activities,
            pseudo_unseen_count=args.pseudo_unseen_count,
            fold_count=args.fold_count,
            seed=42 + split.index,
        )
        split_root = selection_root / f"split_{split.index:03d}"
        candidate_summaries = []
        print(
            f"[{split_position}/{len(splits)}] split={split.index}: "
            f"{len(candidates)} candidates x {len(partitions)} folds",
            flush=True,
        )
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            fold_rows = []
            for fold_index, (pseudo_seen, pseudo_unseen) in enumerate(partitions, start=1):
                fold_seed = 42 + split.index * 1009 + fold_index * 7919
                training, evaluation = build_pseudo_zsl_fold_frames(
                    outer_seen,
                    pseudo_seen=pseudo_seen,
                    pseudo_unseen=pseudo_unseen,
                    samples_per_activity=args.samples_per_activity,
                    seed=fold_seed,
                )
                fold_root = split_root / candidate_id / f"fold_{fold_index:02d}"
                fold_spec = {
                    "split_index": split.index,
                    "candidate_id": candidate_id,
                    "order": candidate["order"],
                    "pseudo_seen": pseudo_seen,
                    "pseudo_unseen": pseudo_unseen,
                    "train_window_indices": training["window_index"].astype(int).tolist(),
                    "evaluation_window_indices": evaluation["window_index"].astype(int).tolist(),
                    "candidate_config_sha256": candidate_sha,
                    "a2_vector_sha256": _sha256(vector_path),
                    "protocol": protocol,
                }
                fold_fingerprint = _fingerprint(fold_spec)
                metrics_path = fold_root / "metrics.json"
                fold_spec_path = fold_root / "fold_spec.json"
                cached = _read_json(metrics_path) if metrics_path.is_file() else None
                if isinstance(cached, Mapping) and cached.get("fingerprint") == fold_fingerprint:
                    metric_row = {name: float(cached[name]) for name in METRICS}
                    print(
                        f"  {candidate_id} fold={fold_index}: reused cached metrics",
                        flush=True,
                    )
                elif args.dry_run:
                    metric_row = {name: 0.0 for name in METRICS}
                    print(
                        f"  {candidate_id} fold={fold_index}: dry-run train={len(training)} "
                        f"eval={len(evaluation)}",
                        flush=True,
                    )
                else:
                    model_path = fold_root / "models"
                    manifest_path = model_path / MANAGER_MANIFEST
                    saved_fold_spec = (
                        _read_json(fold_spec_path) if fold_spec_path.is_file() else None
                    )
                    if model_path.exists() and (
                        not isinstance(saved_fold_spec, Mapping)
                        or saved_fold_spec.get("fingerprint") != fold_fingerprint
                    ):
                        raise RuntimeError(
                            f"Selector model fingerprint mismatch: {model_path}. Use a new "
                            "PAMAP2_ORDER_SELECT_NAME; these models cannot be safely reused."
                        )
                    if model_path.exists() and not manifest_path.is_file():
                        raise RuntimeError(
                            f"Interrupted partial selector model: {model_path}. Move/remove only "
                            "this confirmed partial directory, then resume."
                        )
                    if not model_path.exists():
                        _write_json_atomic(
                            fold_spec_path,
                            {"fingerprint": fold_fingerprint, **fold_spec},
                        )
                    backend_config = {**backend_base, "label_order": candidate["order"]}
                    manager, validation_report = load_or_train_attribute_models(
                        reuse_models=manifest_path.is_file(),
                        model_path=model_path,
                        frame=training,
                        feature_columns=table.feature_columns,
                        attribute_columns=table.attribute_columns,
                        backend="autogluon_soft_oof",
                        backend_config=backend_config,
                        seed=fold_seed,
                    )
                    metrics, _ = evaluate_zsl(
                        manager=manager,
                        frame=evaluation,
                        feature_columns=table.feature_columns,
                        attribute_columns=table.attribute_columns,
                        activity_vectors=table.activity_vectors,
                        unseen_activities=pseudo_unseen,
                        metric="ordinal_probability",
                        attribute_weighting="validation_accuracy",
                        validation_report=validation_report,
                    )
                    metric_row = {
                        "activity_top1_accuracy": metrics["activity"]["top1_accuracy"],
                        "activity_top3_recall": metrics["activity"]["top3_recall"],
                        "activity_macro_f1": metrics["activity"]["macro_f1"],
                    }
                    _write_json_atomic(
                        metrics_path,
                        {"fingerprint": fold_fingerprint, **fold_spec, **metric_row},
                    )
                    print(
                        f"  {candidate_id} fold={fold_index}: "
                        f"top1={metric_row['activity_top1_accuracy']:.4f} "
                        f"top3={metric_row['activity_top3_recall']:.4f} "
                        f"macro_f1={metric_row['activity_macro_f1']:.4f}",
                        flush=True,
                    )
                fold_rows.append({"fold_index": fold_index, **metric_row})
            candidate_summaries.append(
                {
                    **candidate,
                    "folds": fold_rows,
                    "aggregate": aggregate_candidate_folds(
                        fold_rows, stability_penalty=args.stability_penalty
                    ),
                }
            )
        if args.dry_run:
            continue
        selected = select_best_candidate(candidate_summaries)
        split_summary = {
            "split_index": split.index,
            "outer_seen_activities": list(split.seen_activities),
            "final_unseen_activities_excluded_from_selection": list(split.unseen_activities),
            "partitions": [
                {"pseudo_seen": pseudo_seen, "pseudo_unseen": pseudo_unseen}
                for pseudo_seen, pseudo_unseen in partitions
            ],
            "candidates": candidate_summaries,
            "selected_candidate_id": selected["candidate_id"],
            "selected_order": selected["order"],
            "selected_aggregate": selected["aggregate"],
        }
        _write_json_atomic(split_root / "selection_summary.json", split_summary)
        selected_splits[str(split.index)] = {
            "llm_pseudo_zsl": selected["order"],
            "selected_candidate_id": selected["candidate_id"],
            "selected_aggregate": selected["aggregate"],
            "selection_summary_path": str(split_root / "selection_summary.json"),
        }
        for candidate_summary in candidate_summaries:
            aggregate = candidate_summary["aggregate"]
            means = aggregate["metric_means"]
            deviations = aggregate["metric_standard_deviations"]
            report_rows.append(
                {
                    "split_index": split.index,
                    "candidate_id": candidate_summary["candidate_id"],
                    "selected": candidate_summary["candidate_id"] == selected["candidate_id"],
                    "selection_score": aggregate["selection_score"],
                    "mean_utility": aggregate["mean_utility"],
                    "weighted_instability": aggregate["weighted_instability"],
                    **{f"{name}_mean": means[name] for name in METRICS},
                    **{f"{name}_std": deviations[name] for name in METRICS},
                }
            )
        pd.DataFrame(report_rows).to_csv(
            selection_root / "selection_summary.csv", index=False
        )
        print(
            f"  selected={selected['candidate_id']} "
            f"score={selected['aggregate']['selection_score']:.6f}",
            flush=True,
        )

    if args.dry_run:
        print("Dry-run complete; no models or selected-order config were written.", flush=True)
        return
    output = {
        "schema_version": 3,
        "dataset": "pamap2",
        "attribute_set": "A2_refined_per_split",
        "seed": 42,
        "selection_scope": (
            "LLM candidates selected using class-disjoint outer-seen-only pseudo-ZSL; "
            "no final-unseen sensor samples or test metrics"
        ),
        "candidate_config": {"path": str(candidate_path), "sha256": candidate_sha},
        "selection_protocol": protocol,
        "selection_root": str(selection_root),
        "splits": selected_splits,
    }
    _write_json_atomic(output_path, output)
    print(f"Wrote {len(selected_splits)} selected split orders to {output_path}", flush=True)


if __name__ == "__main__":
    main()
