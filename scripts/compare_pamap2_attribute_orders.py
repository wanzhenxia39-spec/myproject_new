#!/usr/bin/env python3
"""Compare PAMAP2 split-specific attribute orders from canonical order files."""

from __future__ import annotations

from itertools import combinations
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ORDER_DIR = PROJECT / "configs/attribute_orders/pamap2"
OUTPUT_DIR = ORDER_DIR / "comparisons"
METHODS = {
    "semantic_dependency_dp": ORDER_DIR / "semantic_dependency_dp_seed42.json",
    "autogluon_seen_only": ORDER_DIR / "autogluon_seen_only_seed42.json",
    "catboost_seen_only": ORDER_DIR / "catboost_seen_only_seed42.json",
    "logistic_regression_seen_only": ORDER_DIR
    / "logistic_regression_seen_only_seed42.json",
}
METRICS = (
    "kendall_tau",
    "pairwise_disagreement_rate",
    "mean_absolute_position_shift",
    "normalized_mean_position_shift",
    "maximum_position_shift",
    "top3_overlap_rate",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def compare_orders(reference: Sequence[str], candidate: Sequence[str]) -> dict[str, Any]:
    left = list(map(str, reference))
    right = list(map(str, candidate))
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValueError("Orders must not contain duplicate attributes")
    if set(left) != set(right) or len(left) != len(right):
        raise ValueError("Orders must contain the same attribute schema")
    size = len(left)
    left_position = {attribute: index for index, attribute in enumerate(left)}
    right_position = {attribute: index for index, attribute in enumerate(right)}
    discordant = 0
    for first, second in combinations(left, 2):
        left_direction = left_position[first] < left_position[second]
        right_direction = right_position[first] < right_position[second]
        discordant += int(left_direction != right_direction)
    pairs = size * (size - 1) // 2
    disagreement = float(discordant / pairs) if pairs else 0.0
    shifts = np.asarray(
        [abs(left_position[attribute] - right_position[attribute]) for attribute in left],
        dtype=float,
    )
    return {
        "attribute_count": size,
        "exact_match": bool(left == right),
        "top1_match": bool(left[0] == right[0]),
        "top3_overlap_count": len(set(left[:3]).intersection(right[:3])),
        "top3_overlap_rate": len(set(left[:3]).intersection(right[:3])) / min(3, size),
        "discordant_pair_count": discordant,
        "attribute_pair_count": pairs,
        "pairwise_disagreement_rate": disagreement,
        "kendall_tau": 1.0 - 2.0 * disagreement,
        "mean_absolute_position_shift": float(shifts.mean()),
        "normalized_mean_position_shift": (
            float(shifts.mean() / (size - 1)) if size > 1 else 0.0
        ),
        "maximum_position_shift": int(shifts.max()) if len(shifts) else 0,
    }


def main() -> int:
    payloads = {method: read_json(path) for method, path in METHODS.items()}
    for method, payload in payloads.items():
        if payload.get("dataset") != "pamap2" or int(payload.get("seed", -1)) != 42:
            raise ValueError(f"Unexpected dataset/seed for {method}: {METHODS[method]}")
        if len(payload.get("splits", {})) != 20:
            raise ValueError(f"Expected 20 splits for {method}: {METHODS[method]}")
    split_sets = [set(map(int, payload["splits"])) for payload in payloads.values()]
    if any(split_set != split_sets[0] for split_set in split_sets[1:]):
        raise ValueError("Order files contain different split indices")

    rows: list[dict[str, Any]] = []
    method_names = list(METHODS)
    for left_method, right_method in combinations(method_names, 2):
        for split_index in sorted(split_sets[0]):
            left = payloads[left_method]["splits"][str(split_index)]["forward"]
            right = payloads[right_method]["splits"][str(split_index)]["forward"]
            rows.append(
                {
                    "left_method": left_method,
                    "right_method": right_method,
                    "split_index": split_index,
                    **compare_orders(left, right),
                }
            )
    per_split = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (left_method, right_method), frame in per_split.groupby(
        ["left_method", "right_method"], sort=False
    ):
        summary: dict[str, Any] = {
            "left_method": left_method,
            "right_method": right_method,
            "split_count": len(frame),
            "exact_match_count": int(frame["exact_match"].sum()),
            "top1_match_count": int(frame["top1_match"].sum()),
            "mean_top3_overlap_count": float(frame["top3_overlap_count"].mean()),
        }
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="raise")
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)

    semantic = summary[summary.left_method == "semantic_dependency_dp"].copy()
    existing = summary[summary.left_method != "semantic_dependency_dp"].copy()
    conclusion = {
        "semantic_vs_classifiers": {
            row["right_method"]: {
                "mean_kendall_tau": row["kendall_tau_mean"],
                "mean_pairwise_disagreement_rate": row[
                    "pairwise_disagreement_rate_mean"
                ],
                "mean_normalized_position_shift": row[
                    "normalized_mean_position_shift_mean"
                ],
                "exact_match_count": int(row["exact_match_count"]),
                "top1_match_count": int(row["top1_match_count"]),
                "mean_top3_overlap_count": row["mean_top3_overlap_count"],
            }
            for row in semantic.to_dict(orient="records")
        },
        "classifier_to_classifier_baseline": {
            "mean_kendall_tau_across_pairs": float(existing.kendall_tau_mean.mean()),
            "mean_pairwise_disagreement_across_pairs": float(
                existing.pairwise_disagreement_rate_mean.mean()
            ),
            "mean_normalized_position_shift_across_pairs": float(
                existing.normalized_mean_position_shift_mean.mean()
            ),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(OUTPUT_DIR / "pairwise_order_comparison_per_split.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "pairwise_order_comparison_summary.csv", index=False)
    write_json(
        OUTPUT_DIR / "semantic_dependency_comparison.json",
        {
            "dataset": "pamap2",
            "seed": 42,
            "order_variant": "forward",
            "note": (
                "Every reverse order is the exact reversal of its forward order, so its "
                "pairwise similarity is identical to the corresponding forward comparison."
            ),
            "methods": {method: str(path.resolve()) for method, path in METHODS.items()},
            "metrics": {
                "kendall_tau": "1 - 2 * discordant_pairs / all_pairs; 1 means identical",
                "pairwise_disagreement_rate": "fraction of attribute pairs ordered differently",
                "normalized_mean_position_shift": "mean absolute rank shift divided by K-1",
                "top3_overlap_rate": "intersection size of the two top-3 sets divided by 3",
            },
            **conclusion,
        },
    )
    print(summary.to_string(index=False))
    print(f"Saved PAMAP2 order comparisons to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
