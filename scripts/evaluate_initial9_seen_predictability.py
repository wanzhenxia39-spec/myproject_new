#!/usr/bin/env python3
"""Replay the existing initial-v1.5 ZSL models on outer 30% Seen validation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd

from zshar.data import load_dataset_table
from zshar.datasets import load_dataset_spec
from zshar.evaluation.attribute_predictability import (
    LEVELS, aggregate_matrices, align_probabilities, audit_partition, score_attribute,
)
from zshar.pipeline.training import split_train_validation
from zshar.splits import load_split_manifest

DATASETS = ["pamap2", "mhealth", "uschad", "wisdm", "dsads"]
NAMES = {"pamap2": "PAMAP2", "mhealth": "MHEALTH", "uschad": "USC-HAD",
         "wisdm": "WISDM", "dsads": "DSADS"}
NEW_RUNS = PROJECT / "runtime/zshar_runs"
OLD_RUNS = Path("/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs")
RUNS = {
    "pamap2": NEW_RUNS / "pamap2/pamap2_zsl_autogluon_default_ensemble_initial_a1_v1_5_independent_t300_random20_eval50_seed42_gpu1_rerun",
    "mhealth": NEW_RUNS / "mhealth/mhealth_zsl_autogluon_default_ensemble_initial_a1_v1_5_independent_t300_random20_eval30_seed42_gpu1_rerun",
    "uschad": OLD_RUNS / "uschad/uschad_zsl_autogluon_default_ensemble_initial9_independent_t300_random20_eval30_seed42",
    "wisdm": OLD_RUNS / "wisdm/wisdm_zsl_autogluon_default_ensemble_initial9_independent_t300_random20_eval30_seed42",
    "dsads": NEW_RUNS / "dsads/dsads_zsl_autogluon_default_ensemble_initial9_independent_t300_random20_eval30_seed42_rerun_20261021",
}
DEFAULT_OUTPUT = Path("/data/wanzhen/experiments/attribute_predictability/initial9_zsl_seen_validation_seed42")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_output(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(Path("/data/wanzhen").resolve()):
        raise ValueError("Model evaluation outputs must be under /data/wanzhen")
    parent = path
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path("/data").stat().st_dev or parent.stat().st_dev == Path("/").stat().st_dev:
        raise ValueError("Output is not on the /data filesystem")
    path.mkdir(parents=True, exist_ok=True)
    return path


def attribute_names():
    return [a["name"] for a in read_json(PROJECT / "configs/attributes/initial_attributes.json")["attributes"]]


def dataset_inputs(ds):
    spec = load_dataset_spec(ds)
    run = RUNS[ds].resolve()
    cfg = read_json(run / "config.json")
    assert cfg["scenario"] == "zsl" and cfg["backend"] == "autogluon" and cfg["seed"] == 42
    assert not cfg.get("attribute_vectors_root") and not cfg.get("attribute_feature_bindings")
    assert spec.attributes_path.name == f"activity_attribute_vectors_initial_{ds}_v1_5.json"
    split_file = PROJECT / f"split/zero_shot_split/{ds}_unseen{spec.unseen_count}_random20_splits_seed42.json"
    splits = load_split_manifest(split_file, spec)
    assert len(splits) == 20 and set(cfg["split_indices"]) == {s.index for s in splits}
    attrs = attribute_names()
    for split in splits:
        directory = run / f"split_{split.index:03d}"
        manifest = read_json(directory / "models/attribute_backend_manifest.json")
        assert manifest["manager_type"] == "independent" and manifest["classifier_backend"] == "autogluon"
        assert manifest["attribute_cols"] == attrs
        assert manifest["backend_config"].get("num_bag_folds", 0) == 0
        saved_split = read_json(directory / "split.json")
        assert set(saved_split["seen_ids"]) == set(split.seen_ids)
        assert set(saved_split["unseen_ids"]) == set(split.unseen_ids)
    return spec, run, splits, split_file


def replay_split(ds, spec, table, run, split, output, resume):
    from autogluon.tabular import TabularPredictor
    directory = output / ds / f"split_{split.index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    model_dir = run / f"split_{split.index:03d}/models"
    manifest_file = model_dir / "attribute_backend_manifest.json"
    manifest = read_json(manifest_file)
    seen = table.frame[table.frame.activity_name.isin(split.seen_activities)]
    train, validation = split_train_validation(seen, seed=42, validation_fraction=0.3)
    assert set(train.activity_name).issubset(split.seen_activities)
    assert set(validation.activity_name).issubset(split.seen_activities)
    assert not set(train.window_index) & set(validation.window_index)
    assert not set(validation.activity_name) & set(split.unseen_activities)
    provenance = {
        "dataset": ds, "split_index": split.index, "model_directory": str(model_dir),
        "manager_sha256": sha256(manifest_file), "attribute_vectors_sha256": sha256(spec.attributes_path),
        "train_window_indices_sha256": hashlib.sha256(train.window_index.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
        "validation_window_indices_sha256": hashlib.sha256(validation.window_index.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
    }
    result_file = directory / "metrics.json"
    if resume and result_file.is_file():
        previous = read_json(result_file)
        if previous["provenance"] != provenance:
            raise ValueError(f"Changed source inputs: {directory}")
        print(f"RESUME {ds} split={split.index}", flush=True)
        return previous["rows"]
    if result_file.exists():
        raise FileExistsError(f"Pass --resume to reuse {directory}")
    partition = pd.concat([
        train[["window_index", "activity_name", "user_id"]].assign(partition="outer_train"),
        validation[["window_index", "activity_name", "user_id"]].assign(partition="outer_validation"),
    ]).rename_axis("dataset_row_index")
    partition.to_csv(directory / "partition.csv", index=True)
    rows, audits = [], {}
    details = validation[["window_index", "activity_name", "user_id"]].copy()
    details.index.name = "dataset_row_index"
    for attr in table.attribute_columns:
        entry = manifest["models"][attr]
        counts = {str(k): int(v) for k, v in train[attr].value_counts().items()}
        if counts != manifest["class_counts"][attr]:
            raise ValueError(f"Saved attribute targets differ from current v1.5: {ds}/{split.index}/{attr}")
        attr_path = model_dir / entry["path"]
        predictor = None
        if entry["backend"] == "constant":
            constant = read_json(attr_path / "constant_model.json")["constant_value"]
            assert set(train[attr]) == {constant}
            probability = pd.DataFrame({constant: np.ones(len(validation))}, index=validation.index)
            prediction = pd.Series(constant, index=validation.index)
            audits[attr] = {"backend": "constant", "outer_overlap": 0,
                           "outer_validation_used_for_autogluon_selection": False}
            best = "constant"
        else:
            predictor = TabularPredictor.load(str(attr_path), verbosity=0)
            y_fit = pd.read_pickle(attr_path / "utils/data/y.pkl")
            y_tune = pd.read_pickle(attr_path / "utils/data/y_val.pkl")
            audits[attr] = audit_partition(y_fit.index, y_tune.index, train.index, validation.index)
            for cached in (y_fit, y_tune):
                expected = predictor.transform_labels(train.loc[cached.index, attr].astype(str))
                if not np.array_equal(expected.to_numpy(), cached.to_numpy()):
                    raise ValueError(f"Stored fit/tuning labels mismatch: {ds}/{split.index}/{attr}")
            metadata = read_json(attr_path / "backend_metadata.json")
            assert metadata["feature_names"] == table.feature_columns
            best = predictor.model_best
            assert best == entry["metadata"]["predictor_model_best"]
            probability = predictor.predict_proba(validation[table.feature_columns])
            # Preserve AutoGluon's saved binary decision threshold, including ties.
            prediction = predictor.predict_from_proba(probability).astype(str)
            audits[attr]["decision_threshold"] = predictor.decision_threshold
            del y_fit, y_tune
        assert probability.index.equals(validation.index) and prediction.index.equals(validation.index)
        score, eod = score_attribute(validation[attr], prediction, probability)
        saved_f1 = manifest["validation_report"]["per_attribute"][attr]["macro_f1"]
        if not np.isclose(score["macro_f1"], saved_f1, atol=1e-10, rtol=0):
            raise ValueError(f"Replayed Macro-F1 differs from saved validation: {ds}/{split.index}/{attr}: {score['macro_f1']} vs {saved_f1}")
        majority = train[attr].value_counts().index[0]
        prior = train[attr].value_counts(normalize=True)
        prior_proba = pd.DataFrame(np.tile(prior.to_numpy(), (len(validation), 1)),
                                  columns=prior.index, index=validation.index)
        baseline, _ = score_attribute(validation[attr], np.repeat(majority, len(validation)), prior_proba)
        row = dict(dataset=ds, split_index=split.index, attribute=attr,
                   n_train=len(train), n_validation=len(validation), training_state_count=len(counts),
                   model_best=best, saved_validation_macro_f1=float(saved_f1), **score,
                   majority_macro_f1=baseline["macro_f1"], training_prior_eod=baseline["eod"],
                   macro_f1_gain_over_majority=score["macro_f1"]-baseline["macro_f1"],
                   eod_reduction_from_prior=baseline["eod"]-score["eod"])
        rows.append(row)
        aligned = align_probabilities(probability)
        details[f"{attr}__true"] = validation[attr]
        details[f"{attr}__pred"] = prediction
        details[f"{attr}__eod"] = eod
        for level in LEVELS:
            details[f"{attr}__p_{level}"] = aligned[level]
        del predictor, probability, prediction, aligned
        gc.collect()
    details.to_csv(directory / "validation_predictions.csv.gz", compression="gzip")
    write_json(directory / "audit.json", audits)
    write_json(result_file, {"provenance": provenance, "rows": rows})
    print(f"DONE {ds} split={split.index} n_val={len(validation)} "
          f"F1={np.mean([r['macro_f1'] for r in rows]):.5f} EOD={np.mean([r['eod'] for r in rows]):.5f}", flush=True)
    return rows


def plot_matrix(frame, title, path, cmap, constant_counts=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(9.7, 6.4), layout="constrained")
        image = ax.imshow(frame.to_numpy(), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(frame.columns)), [NAMES[k] for k in frame.columns])
        ax.set_yticks(range(len(frame)), [k.replace("_", " ") for k in frame.index])
        for i in range(len(frame)):
            for j in range(len(frame.columns)):
                value = frame.iloc[i, j]
                rgba = image.cmap(image.norm(value))
                color = "white" if np.dot(rgba[:3], [.2126, .7152, .0722]) < .5 else "#102030"
                star = "*" if constant_counts is not None and constant_counts.iloc[i, j] > 0 else ""
                text = f"{value:.4f}{star}" if np.isfinite(value) else "pending"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
        ax.set_title(title + "\nInitial v1.5 | outer Seen validation | mean of 20 ZSL splits", pad=12)
        if constant_counts is not None and (constant_counts.to_numpy() > 0).any():
            ax.set_xlabel("* Single-state targets in at least one split; see state_coverage.csv.", labelpad=12, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=.04, pad=.025)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(path.with_suffix("."+extension), dpi=300)
        plt.close(fig)


def write_report(output, matrices, deviations, summary, constant_counts, completion):
    chinese = dict(zip(attribute_names(), ["运动强度", "动作频率", "平滑性", "冲击性", "姿态倾斜度",
                                          "姿态变化幅度", "旋转运动", "平移运动", "运动规律性"]))
    def markdown(frame):
        lines = ["| 属性 | " + " | ".join(NAMES[d] for d in frame.columns) + " |",
                 "|---|" + "---:|" * len(frame.columns)]
        for attr, values in frame.iterrows():
            cells = []
            for ds, value in values.items():
                star = "*" if pd.notna(constant_counts.loc[attr, ds]) and constant_counts.loc[attr, ds] > 0 else ""
                cells.append(f"{value:.4f}{star}" if pd.notna(value) else "待完成")
            lines.append("| " + chinese[attr] + f" (`{attr}`) | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    text = ["# 9维属性的 Seen 验证集可预测性\n",
        "数据集：PAMAP2、MHEALTH、USC-HAD、WISDM、DSADS。使用 initial v1.5 属性原型，seed=42，",
        "每个数据集沿用 20 个 ZSL 类别划分及独立 AutoGluon 属性分类器。\n",
        "完成 split 数：" + "，".join(f"{NAMES[d]} {n}/20" for d, n in completion.items()) + "。\n",
        "## Macro-F1：越高越好\n", markdown(matrices["macro_f1"]),
        "\n## EOD：越低越好\n", markdown(matrices["eod"]),
        "\n两表均为每个属性在 20 个 split 上的等权平均，数值范围 [0,1]。标准差分别见",
        "`macro_f1_std_9x5.csv`、`eod_std_9x5.csv`；标准差为类别划分间的描述性波动。\n",
        "## 统计口径和验证集用途\n",
        "- 每个 split 从 Seen 类窗口中按活动分层留出 30%，其余 70% 为外层训练池。",
        "- 现有 AutoGluon 后端没有把这 30% 作为 tuning_data 传入 fit；它在 70% 内部再留出一部分数据用于模型、集成和二分类阈值选择。",
        "- 逐模型检查缓存行索引和标签，要求内部拟合集与内部选模集的并集恰好为外层 70%，且均不与外层 30% 重合。",
        "- 外层验证准确率曾用于原 ZSL 的活动匹配属性权重，但未用于选择属性分类器；本次直接计算属性预测指标，不执行活动匹配、重新训练、选模或阈值调优。",
        "- 主 Macro-F1 与项目既有实现一致，对真实值与预测值的等级并集求平均，zero_division=0；固定对 Low/Medium/High 三等级求平均的备选矩阵见 `macro_f1_all_levels_mean_9x5.csv`。",
        "- EOD 使用 Low=0、Medium=1、High=2，先对三等级的预测概率加权绝对距离，再对样本求平均，分母固定为 2。缺失训练等级的概率补 0，不压缩等级间距。",
        "- 真实标签来自活动对应的设计属性原型，而非每个 IMU 窗口的独立物理测量标注。\n",
        "## 单一等级与结论边界\n",
        "带 * 的单元格包含至少一个仅有单一训练等级的 split，其高 F1 / 低 EOD 不能单独证明该属性能区分不同等级。具体数量见 `constant_split_count_9x5.csv`。\n"]
    for attr in constant_counts.index:
        for ds in constant_counts.columns:
            n = constant_counts.loc[attr, ds]
            if pd.notna(n) and n > 0:
                text.append(f"- {NAMES[ds]} / {chinese[attr]}：{int(n)} 个 split 为单一训练等级。")
    text += ["\n训练集多数类预测的 Macro-F1、训练等级分布的 EOD 均单独导出，便于与无信号基线比较。",
        "结果支持对已有 Seen 活动语义属性标签的窗口级预测能力；不直接证明 Unseen 活动、独立受试者或独立采集记录上的泛化。现有窗口划分可能共享受试者和重叠原始记录。\n",
        "## 文件\n",
        "- `attribute_predictability_9x5.xlsx`：两项主指标、标准差、备选口径、基线及明细。",
        "- `macro_f1_heatmap_9x5.{png,pdf,svg}` 和 `eod_heatmap_9x5.{png,pdf,svg}`：两张热力图。",
        "- `per_split_attribute_metrics.csv`：100 个 split × 9 属性的逐 split 指标。",
        "- `<dataset>/split_<index>/validation_predictions.csv.gz`：验证窗口、真值、预测等级、三等级概率和逐样本 EOD。",
        "- 各 split 的 `partition.csv`、`audit.json`、`metrics.json`：划分、隔离检查和指标来源。",
        "- `protocol.json`、`code_snapshot/` 和 `README.md`：协议、代码快照和重跑说明。\n"]
    (output / "REPORT_zh.md").write_text("\n".join(text) + "\n")


def summarize(rows, output, datasets):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    attributes = attribute_names()
    frame.to_csv(output / "per_split_attribute_metrics.csv", index=False)
    metrics = ["macro_f1", "eod", "macro_f1_all_levels", "majority_macro_f1",
               "training_prior_eod", "macro_f1_gain_over_majority", "eod_reduction_from_prior"]
    matrices, deviations = {}, {}
    for metric in metrics:
        mean, std, count = aggregate_matrices(frame, attributes, datasets, metric)
        for suffix, matrix in [("mean", mean), ("std", std), ("count", count)]:
            matrix.to_csv(output / f"{metric}_{suffix}_9x5.csv", float_format="%.10f")
        matrices[metric] = mean
        deviations[metric] = std
    constant_counts = frame.assign(constant=frame.training_state_count.eq(1)).groupby(
        ["attribute", "dataset"])["constant"].sum().unstack("dataset").reindex(index=attributes, columns=datasets)
    constant_counts.to_csv(output / "constant_split_count_9x5.csv")
    for metric, cmap, title in [("macro_f1", "YlGnBu", "Attribute prediction Macro-F1 (higher is better)"),
                                ("eod", "YlOrRd", "Expected ordinal distance (lower is better)")]:
        plot_matrix(matrices[metric], title, output / f"{metric}_heatmap_9x5", cmap, constant_counts)
    coverage = frame.groupby(["attribute", "dataset"]).agg(
        mean_train_states=("training_state_count", "mean"),
        mean_validation_states=("validation_state_count", "mean"),
        min_validation_states=("validation_state_count", "min"),
        split_count=("split_index", "count"),
    )
    coverage.to_csv(output / "state_coverage.csv")
    dataset_summary = frame.groupby("dataset").agg(
        mean_attribute_macro_f1=("macro_f1", "mean"), mean_attribute_eod=("eod", "mean"),
        mean_majority_macro_f1=("majority_macro_f1", "mean"), mean_training_prior_eod=("training_prior_eod", "mean"),
    ).reindex(datasets)
    dataset_summary.to_csv(output / "dataset_summary.csv")
    completion = {ds: int(frame[frame.dataset.eq(ds)].split_index.nunique()) for ds in datasets}
    write_json(output / "status.json", {"completed_splits": completion,
               "complete": all(n == 20 for n in completion.values()), "attribute_split_rows": len(frame)})
    write_report(output, matrices, deviations, dataset_summary, constant_counts, completion)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        pass
    else:
        with pd.ExcelWriter(output / "attribute_predictability_9x5.xlsx") as writer:
            for metric, matrix in matrices.items():
                matrix.rename(columns=NAMES).to_excel(writer, sheet_name=metric[:31])
            for metric in ("macro_f1", "eod"):
                deviations[metric].rename(columns=NAMES).to_excel(writer, sheet_name=metric + "_std")
            constant_counts.rename(columns=NAMES).to_excel(writer, sheet_name="constant_split_count")
            coverage.to_excel(writer, sheet_name="state_coverage")
            frame.to_excel(writer, sheet_name="per_split", index=False)
            dataset_summary.to_excel(writer, sheet_name="dataset_summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = validate_output(args.output)
    assert len(attribute_names()) == 9
    inputs = {}
    versions = {key: importlib.metadata.version(key) for key in ["numpy", "pandas", "scikit-learn", "autogluon.tabular"]}
    provenance = {"protocol": "existing_initial9_v1.5_independent_AutoGluon_ZSL_outer_seen_validation",
        "seed": 42, "outer_validation_fraction": .3, "levels": list(LEVELS), "eod_denominator": 2,
        "macro_f1_label_policy": "union of ground-truth and predicted states, matching existing project",
        "secondary_f1_policy": "all declared Low/Medium/High states, zero_division=0",
        "aggregation": "equal-weight arithmetic mean over 20 class splits; std ddof=1",
        "model_selection": "AutoGluon selects models/ensembles and any binary threshold on its INNER holdout. The OUTER 30% is not supplied to fit(). No model is selected or changed by this analysis.",
        "scope": "Seen-window held-out performance for activity-derived attribute labels, not unseen-activity transfer or cross-subject performance. Window splits may share subjects and overlapping source recordings.",
        "no_training": True, "no_unseen_test_evaluation": True, "versions": versions, "datasets": {}}
    for ds in args.datasets:
        spec, run, splits, split_file = dataset_inputs(ds)
        inputs[ds] = (spec, run, splits)
        provenance["datasets"][ds] = {"run": str(run), "split_manifest": str(split_file),
            "split_manifest_sha256": sha256(split_file), "vectors": str(spec.attributes_path),
            "vectors_sha256": sha256(spec.attributes_path), "split_count": len(splits)}
    write_json(output / "protocol.json", provenance)
    if args.check:
        print(json.dumps(provenance, indent=2)); return
    import torch
    torch.set_num_threads(4)
    rows = []
    for ds in args.datasets:
        spec, run, splits = inputs[ds]
        table = load_dataset_table(spec)
        assert table.attribute_columns == attribute_names()
        selected = splits[:args.max_splits] if args.max_splits else splits
        print(f"DATASET {ds} rows={len(table.frame)} features={len(table.feature_columns)} splits={len(selected)}", flush=True)
        for split in selected:
            started = time.monotonic()
            rows.extend(replay_split(ds, spec, table, run, split, output, args.resume))
            print(f"ELAPSED {ds} split={split.index} seconds={time.monotonic()-started:.1f}", flush=True)
        del table
        summarize(rows, output, args.datasets)
        gc.collect()
    print("FINISHED " + str(output), flush=True)


if __name__ == "__main__":
    main()
