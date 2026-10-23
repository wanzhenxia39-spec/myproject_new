#!/usr/bin/env python3
"""Run uncalibrated PAMAP2 initial-A1 GZSL on a selected physical GPU.

The experiment reuses the seed-42 20-split GZSL protocol, fixed evaluation
samples, pre-refinement initial A1 v1.5 attribute vectors, and independent
AutoGluon default ensembles from the matched calibrated experiment. It
deliberately disables every seen/unseen bias correction: predictions use raw
compatibility scores with no temperature softmax and no seen-class penalty.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT / "scripts/run_pamap2_a2_gzsl_autogluon_default_ensemble.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "pamap2_gzsl_default_ensemble_base", BASE_SCRIPT
)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

RUN_ROOT = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
RUN_NAME = (
    "pamap2_gzsl_autogluon_default_ensemble_initial_a1_v1_5_"
    "independent_no_calibration_fixed_t300_"
    "random20_eval20_seen20_seed42_gpu0"
)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def experiment_config(args: argparse.Namespace, selected: Sequence[int]) -> Any:
    """Build a strict raw-score identity configuration with no calibration."""
    configured = base.experiment_config(args, selected)
    return replace(
        configured,
        run_name=args.run_name,
        gzsl_bias_correction="none",
        calibration_mode="fixed",
        temperature=1.0,
        gamma=0.0,
        gamma_grid=None,
        calibration_temperature_grid=(1.0,),
    )


def summarize(config: Any, selected: Sequence[int]) -> Path:
    run_dir = Path(config.run_root) / base.DATASET / str(config.run_name)
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(summary)

    frame = pd.read_csv(summary).sort_values("split_index")
    expected = set(map(int, selected))
    if len(frame) != len(expected) or set(frame.split_index.astype(int)) != expected:
        raise ValueError(f"Incomplete GZSL summary: {summary}")
    if set(frame.backend.astype(str)) != {"autogluon"}:
        raise ValueError(f"Unexpected backend in {summary}")
    if "attribute_order_path" in frame and frame.attribute_order_path.notna().any():
        raise ValueError("Independent run unexpectedly recorded an attribute order")
    if set(frame.gzsl_bias_correction.astype(str)) != {"none"}:
        raise ValueError(f"Unexpected bias correction in {summary}")
    if set(frame.calibration_mode.astype(str)) != {"fixed"}:
        raise ValueError("No-calibration run must use fixed identity parameters")
    temperatures = pd.to_numeric(frame.temperature, errors="raise")
    gammas = pd.to_numeric(frame.gamma, errors="raise")
    if not temperatures.eq(1.0).all() or not gammas.eq(0.0).all():
        raise ValueError("No-calibration run requires temperature=1 and gamma=0")

    rows: list[dict[str, Any]] = []
    for metric in base.REPORT_METRICS:
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
    report.to_csv(run_dir / "gzsl_report.csv", index=False)
    write_json(
        run_dir / "gzsl_report.json",
        {
            "dataset": base.DATASET,
            "scenario": "gzsl",
            "backend": "autogluon",
            "classifier": "independent AutoGluon default ensemble per attribute",
            "attribute_set": base.ATTRIBUTE_SET,
            "attribute_vectors_root": None,
            "attribute_vectors_path": str(base.A1_VECTORS),
            "attribute_order": None,
            "attribute_dependency": "independent",
            "split_manifest": str(base.SPLITS),
            "sample_manifest": str(base.SAMPLES),
            "calibration": (
                "disabled; raw compatibility scores with no temperature softmax "
                "and no seen-class penalty"
            ),
            "calibration_mode": "fixed",
            "bias_correction": "none",
            "fixed_temperature": 1.0,
            "fixed_gamma": 0.0,
            "split_count": len(expected),
            "metrics": rows,
        },
    )
    print("\n" + report.to_string(index=False), flush=True)
    print(f"GZSL report saved to: {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("check", "train-test", "report"), default="train-test"
    )
    parser.add_argument("--gpu-id", type=int, choices=(0, 1), default=0)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--split-indices", help="Optional comma-separated split subset")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_cpus <= 0 or args.time_limit != 300:
        raise ValueError("This matched protocol requires positive CPUs and --time-limit 300")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    split_payload = base.read_json(base.SPLITS)
    all_indices = [int(item["split_index"]) for item in split_payload.get("splits", [])]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct PAMAP2 GZSL splits")
    selected = base.matched.selected_indices(args, all_indices)
    spec, _, verified = base.preflight(selected)
    if verified != all_indices:
        raise ValueError("GZSL split manifest changed during protocol validation")

    config = experiment_config(args, selected)
    run_dir = base.validate_existing_run(config)
    print(
        "PAMAP2 initial-A1 v1.5 GZSL AutoGluon default ensemble\n"
        "  attribute dependency: independent (no attribute order)\n"
        "  classifier: AutoGluon default model suite + weighted ensemble\n"
        "  calibration: disabled (no calibration folds)\n"
        "  bias correction: none (raw scores; temperature=1, gamma=0)\n"
        f"  physical GPU: {args.gpu_id} (logical device 0)\n"
        f"  split manifest: {base.SPLITS}\n"
        f"  sample manifest: {base.SAMPLES}\n"
        f"  selected splits: {selected}\n"
        f"  initial A1 vectors: {base.A1_VECTORS}\n"
        f"  run: {run_dir}",
        flush=True,
    )
    if args.mode == "check":
        print("Check only: no classifier or calibration model was trained.", flush=True)
        return 0
    if args.mode == "report":
        summarize(config, selected)
        return 0

    base.preserve_partial_models(run_dir, resume=args.resume)
    from zshar.pipeline.runner import run_experiment

    output = run_experiment(spec, config)
    print(
        "Completed PAMAP2 initial-A1 GZSL AutoGluon default ensemble "
        f"(no calibration): {output}",
        flush=True,
    )
    summarize(config, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
