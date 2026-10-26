#!/usr/bin/env python3
"""PAMAP2/MHEALTH GZSL: revised initial9, independent AutoGluon default ensembles.

Temperature is selected by pseudo-GZSL validation NLL using outer-Seen data
only. Gamma is always zero; no seen penalty or AUSUC gamma sweep is performed.
All generated files must physically reside on /data under /data/wanzhen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
OUTPUT = Path("/data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42")
VECTORS = PROJECT / "configs/attributes_new"
SPLIT_ROOT = PROJECT / "split/gzsl_20"
RUN_NAME = "initial9_new_autogluon_default_temperature_only_cal3_t300_seed42"
TEMPERATURES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MANIFEST = "attribute_backend_manifest.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(value):
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(normalize(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    return str(value) if isinstance(value, Path) else value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(normalize(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_output(path):
    """Resolve symlinks and check the filesystem before creating output paths."""
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(Path("/data/wanzhen").resolve()):
        raise ValueError("All experiment outputs must be under /data/wanzhen")
    existing = path
    while not existing.exists():
        existing = existing.parent
    if (existing.stat().st_dev != Path("/data").stat().st_dev
            or existing.stat().st_dev == Path("/").stat().st_dev):
        raise ValueError("Output must physically reside on the data filesystem")
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_runtime(root, gpu_id, num_cpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTHONUNBUFFERED"] = "1"
    for variable, folder in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"),
                             ("XDG_CACHE_HOME", "xdg"), ("TORCH_HOME", "torch"),
                             ("CUDA_CACHE_PATH", "cuda")):
        os.environ[variable] = str(data_output(root / "cache" / folder))
    # This also covers direct Python invocation; the shell sets it at startup.
    sys.pycache_prefix = str(data_output(root / "cache/pycache"))
    os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(num_cpus)


def experiment_config(args, selected, output):
    from zshar.pipeline.runner import ExperimentConfig

    return ExperimentConfig(
        dataset=args.dataset, scenario="gzsl", backend="autogluon",
        backend_config={
            "time_limit": 300, "presets": "medium_quality", "num_gpus": 1,
            "num_cpus": args.num_cpus, "num_bag_folds": 0, "num_stack_levels": 0,
            "fit_weighted_ensemble": True, "model_suite": "default",
            "lightgbm_device": "cuda",
            "gpu_model_types": ["NN_TORCH", "FASTAI", "CAT", "XGB"], "verbosity": 2,
        },
        seed=42,
        split_manifest=str(SPLIT_ROOT / f"{args.dataset}_unseen4_random20_gzsl_splits_seed42.json"),
        sample_manifest=str(SPLIT_ROOT / f"{args.dataset}_unseen4_random20_gzsl_eval20_seen20_samples_seed42.json"),
        split_indices=set(selected), samples_per_activity=20,
        prototype_metric="ordinal_probability", attribute_weighting="validation_accuracy",
        gzsl_bias_correction="temperature_only", gamma=0.0, gamma_grid=None,
        temperature=1.0, calibration_mode="grid", calibration_class_folds=3,
        calibration_pseudo_unseen_count=4, calibration_samples_per_activity=3,
        calibration_temperature_grid=TEMPERATURES,
        run_root=str(output / "runs"), run_name=RUN_NAME, reuse_models=args.resume,
    )


def frame_signature(frame):
    return hashlib.sha256(frame.window_index.to_numpy(dtype="int64").tobytes()).hexdigest()


def validate_model_manifest(model_dir, frame, attributes, seed):
    """Reject saved models fitted with different targets, including old regularity."""
    from zshar.pipeline.training import split_train_validation

    saved = read(model_dir / MANIFEST)
    if (saved["manager_type"] != "independent"
            or saved["classifier_backend"] != "autogluon"
            or saved["attribute_cols"] != attributes):
        raise ValueError(f"Incompatible classifier/schema: {model_dir}")
    fit, _ = split_train_validation(frame, seed=seed)
    counts = {a: {str(k): int(v) for k, v in fit[a].value_counts().items()} for a in attributes}
    if saved["class_counts"] != counts:
        raise ValueError(f"Saved classifier targets differ from revised initial9: {model_dir}")


def preflight(args, config):
    from zshar.data import load_activity_vectors, load_dataset_table
    from zshar.datasets import load_dataset_spec
    from zshar.pipeline.calibration import build_calibration_fold_frames, select_pseudo_unseen_partitions
    from zshar.splits import load_sample_manifest, load_split_manifest, sample_evaluation_rows

    vector_path = VECTORS / f"activity_attribute_vectors_initial_{args.dataset}_v1_5.json"
    ontology_path = VECTORS / "initial_attributes.json"
    profile_path = PROJECT / f"configs/datasets/{args.dataset}.json"
    spec = load_dataset_spec(args.dataset, profile_path)
    spec = replace(spec, attributes_path=vector_path.resolve(),
                   raw={**spec.raw, "attributes": str(vector_path.resolve())})
    vectors, attributes = load_activity_vectors(vector_path)
    expected = [a["name"] for a in read(ontology_path)["attributes"]]
    if len(attributes) != 9 or attributes != expected or set(vectors) != set(spec.activity_names):
        raise ValueError("Revised vectors must match the ordered initial9 ontology and all 12 activities")
    if any(v not in {"Low", "Medium", "High"} for row in vectors.values() for v in row.values()):
        raise ValueError("Invalid initial9 attribute levels")
    splits_json, samples_json = read(config.split_manifest), read(config.sample_manifest)
    for payload in (splits_json, samples_json):
        if (payload["dataset"] != args.dataset or payload["seed"] != 42
                or payload["evaluation_protocol"] != "gzsl"):
            raise ValueError("Wrong dataset, seed or evaluation protocol in GZSL manifests")
    if (splits_json["num_splits"] != 20 or splits_json["unseen_count"] != 4
            or splits_json["training_label_space"] != "seen_only"
            or splits_json["candidate_label_space"] != "seen_union_unseen"
            or samples_json["seen_samples_per_activity"] != 20
            or samples_json["unseen_samples_per_activity"] != 20):
        raise ValueError("Expected 20 splits, Seen-only training and 20 Seen/Unseen test rows/class")
    splits = load_split_manifest(config.split_manifest, spec)
    all_indices = [s.index for s in splits]
    if len(all_indices) != 20 or len(set(all_indices)) != 20:
        raise ValueError("Expected exactly 20 distinct splits")
    if not config.split_indices or not config.split_indices.issubset(all_indices):
        raise ValueError("Selected split indices must be a nonempty subset of the requested manifest")
    sample_ids = load_sample_manifest(config.sample_manifest)
    if set(sample_ids) != set(all_indices):
        raise ValueError("Split and sample manifest indices differ")
    table = load_dataset_table(spec)
    run = Path(config.run_root) / args.dataset / config.run_name
    audits = []
    for split, raw in zip(splits, splits_json["splits"]):
        if split.index not in config.split_indices:
            continue
        # The loader derives Seen classes from Unseen IDs; also validate the explicit JSON Seen IDs.
        if (len(split.seen_ids) != 8 or len(split.unseen_ids) != 4
                or set(map(int, raw["seen_ids"])) != set(split.seen_ids)
                or set(map(int, raw["unseen_ids"])) != set(split.unseen_ids)):
            raise ValueError(f"Invalid 8 Seen + 4 Unseen partition: {split.index}")
        groups = samples_json["samples"][str(split.index)]
        if (len(groups) != 12 or any(len(ids) != 20 or len(set(ids)) != 20 for ids in groups.values())
                or len(sample_ids[split.index]) != 240):
            raise ValueError(f"Invalid fixed evaluation windows: {split.index}")
        evaluation = sample_evaluation_rows(table.frame, split, scenario="gzsl",
            samples_per_activity=20, seed=42, manifest_ids=sample_ids[split.index])
        if (len(evaluation) != 240 or not evaluation.activity_name.value_counts().eq(20).all()
                or set(evaluation.window_index) != sample_ids[split.index]):
            raise ValueError(f"Fixed windows do not resolve to 12 x 20 rows: {split.index}")
        # MHEALTH JSON uses legacy names. Resolve each group by actual window labels.
        indexed = evaluation.set_index("window_index")
        resolved = [set(indexed.loc[list(map(int, ids)), "activity_name"]) for ids in groups.values()]
        if any(len(names) != 1 for names in resolved) or len(set.union(*resolved)) != 12:
            raise ValueError(f"Sample groups do not map one-to-one to activities: {split.index}")
        training = table.frame[table.frame.activity_name.isin(split.seen_activities)
                               & ~table.frame.window_index.isin(sample_ids[split.index])].copy()
        if (set(training.activity_name) != set(split.seen_activities)
                or set(training.window_index) & sample_ids[split.index]):
            raise ValueError(f"Invalid or overlapping outer Seen training set: {split.index}")
        model_frames = [(run / f"split_{split.index:03d}/models", training, 42)]
        fold_audits = []
        partitions = select_pseudo_unseen_partitions(split.seen_activities,
            pseudo_unseen_count=4, fold_count=3, seed=42 + split.index)
        for fold, (pseudo_seen, pseudo_unseen) in enumerate(partitions, 1):
            fit, validation = build_calibration_fold_frames(training, pseudo_seen=pseudo_seen,
                pseudo_unseen=pseudo_unseen, samples_per_activity=3,
                seed=42 + split.index + fold * 7919)
            model_frames.append((run / f"split_{split.index:03d}/calibration/fold_{fold:02d}/models",
                                 fit, 42 + split.index + fold))
            fold_audits.append({"fold": fold, "pseudo_seen": pseudo_seen, "pseudo_unseen": pseudo_unseen,
                "train_ids_sha256": frame_signature(fit), "validation_ids_sha256": frame_signature(validation)})
        for model_dir, frame, seed in model_frames:
            if (model_dir / MANIFEST).is_file():
                validate_model_manifest(model_dir, frame, attributes, seed)
        audits.append({"split_index": split.index, "train_rows": len(training), "test_rows": len(evaluation),
            "train_ids_sha256": frame_signature(training), "test_ids_sha256": frame_signature(evaluation),
            "calibration_folds": fold_audits})
    files = (vector_path, ontology_path, profile_path, Path(config.split_manifest), Path(config.sample_manifest))
    protocol = {"dataset": args.dataset, "vectors": str(vector_path.resolve()), "attributes": attributes,
        "input_sha256": {str(p.resolve()): sha(p) for p in files},
        "feature_files": {str(p.resolve()): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                          for p in (spec.features_path, spec.feature_names_path, spec.labels_path)},
        "config": normalize(asdict(config)), "split_audits": audits,
        "temperature_selection": "3-fold outer-Seen-only pseudo-GZSL; minimize mean validation NLL",
        "temperature_effect": "softmax(activity_scores/T) preserves rankings; gamma=0; no AUSUC sweep"}
    protocol["config"]["reuse_models"] = False
    return spec, run, protocol


def validate_existing_run(run, protocol, resume, mode):
    saved = run / "input_protocol.json"
    if saved.is_file():
        if read(saved) != normalize(protocol):
            raise ValueError(f"Inputs/configuration changed; choose a fresh output root: {saved}")
        if mode == "train-test" and not resume:
            raise FileExistsError("Run already exists; use --resume")
    elif (run / "config.json").exists() or list(run.glob("split_*")):
        raise ValueError("Existing models lack the revised-vector input signature; use a fresh output root")


def preserve_partial_models(run):
    for path in list(run.glob("split_*/models")) + list(run.glob("split_*/calibration/fold_*/models")):
        if (path / MANIFEST).is_file():
            continue
        backup = path.with_name("models.incomplete")
        index = 1
        while backup.exists():
            backup = path.with_name(f"models.incomplete_{index}")
            index += 1
        shutil.move(str(path), str(backup))
        print(f"Preserved interrupted model directory: {backup}", flush=True)


def summarize(run, config, protocol):
    import pandas as pd

    frame = pd.read_csv(run / "summary.csv")
    if len(frame) != len(config.split_indices) or set(frame.split_index) != config.split_indices:
        raise ValueError("Incomplete or duplicate split summary")
    if (not frame.gamma.eq(0.0).all() or set(frame.gzsl_bias_correction) != {"temperature_only"}
            or set(frame.backend) != {"autogluon"}
            or not frame.attribute_count.eq(9).all()
            or set(frame.attribute_vectors_sha256) != {sha(protocol["vectors"])}):
        raise ValueError("Summary violates revised initial9 / temperature-only invariants")
    nlls = []
    for index in frame.split_index:
        metrics = read(run / f"split_{index:03d}/metrics.json")
        calibration = metrics["calibration"]
        if (metrics["gamma"] != 0 or calibration["best_gamma"] != 0
                or calibration["selection_metric"] != "mean_validation_negative_log_likelihood"):
            raise ValueError("Temperature must be selected by validation NLL with gamma fixed at zero")
        nlls.append(calibration["best_mean_nll"])
    frame["calibration_validation_nll"] = nlls
    names = ("seen_accuracy", "unseen_accuracy", "harmonic_mean", "overall_accuracy",
             "activity_top1_accuracy", "activity_top3_recall", "activity_macro_f1",
             "temperature", "gamma", "calibration_validation_nll")
    rows = []
    for name in names:
        values = pd.to_numeric(frame[name], errors="raise")
        rows.append({"metric": name, "split_count": len(values), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max())})
    frame.to_csv(run / "gzsl_per_split.csv", index=False)
    pd.DataFrame(rows).to_csv(run / "gzsl_report.csv", index=False)
    write(run / "gzsl_report.json", {"dataset": config.dataset, "protocol": protocol,
                                    "metrics": rows, "ausuc": "not computed; no gamma sweep"})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"REPORT: {run}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("pamap2", "mhealth"))
    parser.add_argument("mode", nargs="?", choices=("check", "train-test", "report"), default="check")
    parser.add_argument("--gpu-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--split-indices", help="Optional comma-separated subset; default all 20 JSON splits")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.num_cpus <= 0:
        raise ValueError("--num-cpus must be positive")
    output = data_output(args.output_root)
    dataset_output = data_output(output / args.dataset)
    configure_runtime(dataset_output, args.gpu_id, args.num_cpus)
    # Keep this descriptor open throughout preflight, training and reporting.
    with (dataset_output / "experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another {args.dataset} experiment/check is already running") from exc
        split_file = SPLIT_ROOT / f"{args.dataset}_unseen4_random20_gzsl_splits_seed42.json"
        available = [int(s["split_index"]) for s in read(split_file)["splits"]]
        selected = available if args.split_indices is None else sorted(
            {int(i.strip()) for i in args.split_indices.split(",") if i.strip()})
        config = experiment_config(args, selected, output)
        spec, run, protocol = preflight(args, config)
        data_output(run)
        validate_existing_run(run, protocol, args.resume, args.mode)
        print(json.dumps({"mode": args.mode, "physical_gpu": args.gpu_id, "logical_gpu": 0,
            "vectors": str(spec.attributes_path), "splits": selected,
            "split_manifest": config.split_manifest, "sample_manifest": config.sample_manifest,
            "backend": config.backend_config, "bias_correction": "temperature_only", "gamma": 0,
            "temperature_selection": protocol["temperature_selection"], "run": str(run)},
            ensure_ascii=False, indent=2), flush=True)
        if args.mode == "check":
            write(dataset_output / "preflight.json", protocol)
            print("CHECK PASSED: no model training or prediction was performed.", flush=True)
            return 0
        if args.mode == "report":
            summarize(run, config, protocol)
            return 0
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly the requested physical GPU as logical cuda:0")
        write(run / "input_protocol.json", protocol)
        snapshots = data_output(run / "input_snapshot")
        for source in protocol["input_sha256"]:
            shutil.copy2(source, snapshots / Path(source).name)
        shutil.copy2(__file__, snapshots / Path(__file__).name)
        preserve_partial_models(run)
        from zshar.pipeline.runner import run_experiment
        run_experiment(spec, config)
        summarize(run, config, protocol)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
