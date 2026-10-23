#!/usr/bin/env python3
"""Run PAMAP2 semantic-v3 A2 GZSL calibration ablations on GPU 0.

The classifier and GZSL protocol match the maintained calibrated PAMAP2 run:
AutoGluon soft-OOF probability chains, split-specific refined A2 vectors, and
reused seen-only CatBoost forward orders.  Only the calibration component is
changed: temperature-only, seen-penalty-only, or identity/no-calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = (
    PROJECT / "scripts/run_pamap2_a2_gzsl_autogluon_soft_oof_reuse_zsl_order.py"
)
BASE_SPEC = importlib.util.spec_from_file_location("pamap2_calibrated_gzsl_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

OLD_GZSL_ROOT = Path("/home/wanzhen/myproject/split/gzsl_20")
base.SPLITS = OLD_GZSL_ROOT / "pamap2_unseen4_random20_gzsl_splits_seed42.json"
base.SAMPLES = (
    OLD_GZSL_ROOT
    / "pamap2_unseen4_random20_gzsl_eval20_seen20_samples_seed42.json"
)

RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
VARIANTS = ("temperature_only", "seen_penalty_only", "no_calibration")
RUN_NAMES = {
    variant: (
        "pamap2_gzsl_autogluon_soft_oof_a2_semantic_v3_"
        "reuse_zsl_catboost_seen_only_forward_ablation_"
        f"{variant}_bag2_random20_eval20_seen20_seed42_gpu0"
    )
    for variant in VARIANTS
}
REPORT_METRICS = (
    "seen_accuracy",
    "unseen_accuracy",
    "harmonic_mean",
    "overall_accuracy",
    "activity_top1_accuracy",
    "activity_top3_recall",
    "activity_macro_f1",
    "ausuc",
    "calibration_validation_ausuc",
    "temperature",
    "gamma",
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, target)


def experiment_config(args: argparse.Namespace, selected: Sequence[int]) -> Any:
    configured = base.experiment_config(args, selected)
    common = {
        "run_name": args.run_name or RUN_NAMES[args.variant],
        "gamma": 0.0,
        "temperature": 1.0,
    }
    if args.variant == "temperature_only":
        return replace(
            configured,
            **common,
            gzsl_bias_correction="temperature_only",
            calibration_mode="grid",
            gamma_grid=None,
        )
    if args.variant == "seen_penalty_only":
        return replace(
            configured,
            **common,
            gzsl_bias_correction="seen_penalty_only",
            calibration_mode="grid",
            gamma_grid=base.gamma_grid(),
        )
    return replace(
        configured,
        **common,
        gzsl_bias_correction="none",
        calibration_mode="fixed",
        gamma_grid=None,
    )


def calibration_description(variant: str) -> str:
    if variant == "temperature_only":
        return (
            "temperature-only: select temperature by outer-seen-only pseudo-GZSL "
            "validation NLL; seen penalty removed (gamma=0)"
        )
    if variant == "seen_penalty_only":
        return (
            "seen-penalty-only: select gamma by outer-seen-only pseudo-GZSL; "
            "temperature scaling/softmax removed"
        )
    return (
        "no-calibration: raw compatibility scores; no calibration folds; "
        "temperature=1 and gamma=0"
    )


def summarize(config: Any, selected: Sequence[int], variant: str) -> Path:
    run_dir = Path(config.run_root) / base.DATASET / str(config.run_name)
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    frame = pd.read_csv(summary).sort_values("split_index")
    expected = set(map(int, selected))
    if len(frame) != len(expected) or set(frame.split_index.astype(int)) != expected:
        raise ValueError(f"Incomplete GZSL summary: {summary}")
    if set(frame.backend.astype(str)) != {"autogluon_soft_oof"}:
        raise ValueError(f"Unexpected backend in {summary}")

    rows: list[dict[str, Any]] = []
    for metric in REPORT_METRICS:
        if metric not in frame:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "split_count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(run_dir / "gzsl_ablation_report.csv", index=False)
    write_json(
        run_dir / "gzsl_ablation_report.json",
        {
            "dataset": base.DATASET,
            "scenario": "gzsl",
            "backend": "autogluon_soft_oof",
            "attribute_set": base.ATTRIBUTE_SET,
            "attribute_vectors_root": str(base.A2_ROOT),
            "split_manifest": str(base.SPLITS),
            "sample_manifest": str(base.SAMPLES),
            "attribute_order": "reused ZSL seen-only CatBoost forward order",
            "chain_features": (
                f"{config.backend_config['num_bag_folds']}-fold bagged OOF "
                "probabilities plus normalized entropy"
            ),
            "ablation_variant": variant,
            "calibration": calibration_description(variant),
            "calibration_mode": config.calibration_mode,
            "bias_correction": config.gzsl_bias_correction,
            "fixed_temperature": float(config.temperature),
            "fixed_gamma": float(config.gamma),
            "split_count": len(expected),
            "metrics": rows,
        },
    )
    print("\n" + report.to_string(index=False), flush=True)
    print(f"GZSL ablation report saved to: {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test", "report"), default="train-test"
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--num-bag-folds", type=int, default=2)
    parser.add_argument("--split-indices", help="Optional comma-separated split subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_id not in (0, 1):
        raise ValueError("This ablation queue is intentionally pinned to physical GPU 0")
    if args.num_cpus != 8 or args.time_limit != 300 or args.num_bag_folds != 2:
        raise ValueError("This matched protocol requires num-cpus=8, time-limit=300, bag2")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    split_payload = read_json(base.SPLITS)
    all_indices = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    selected = base.selected_indices(args, all_indices)
    spec, splits, verified = base.preflight(selected)
    if verified != all_indices:
        raise ValueError("GZSL split manifest changed during protocol validation")
    base.validate_reused_order(splits, selected)
    config = experiment_config(args, selected)
    run_dir = base.validate_existing_run(config)

    print(
        "PAMAP2 semantic-v3 A2 GZSL AutoGluon soft-OOF calibration ablation\n"
        f"  variant: {args.variant}\n"
        f"  calibration: {calibration_description(args.variant)}\n"
        "  order: reused ZSL seen-only CatBoost forward order\n"
        f"  chain: {args.num_bag_folds}-fold bagged OOF probabilities + normalized entropy\n"
        f"  split manifest: {base.SPLITS}\n"
        f"  sample manifest: {base.SAMPLES}\n"
        f"  A2 root: {base.A2_ROOT}\n"
        f"  selected splits: {selected}\n"
        f"  run: {run_dir}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier or calibration model was trained.", flush=True)
        return 0
    if args.mode == "report":
        summarize(config, selected, args.variant)
        return 0

    base.preserve_partial_models(run_dir, resume=args.resume)
    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, config)
    print(f"Completed PAMAP2 GZSL ablation {args.variant}: {output}", flush=True)
    summarize(config, selected, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
