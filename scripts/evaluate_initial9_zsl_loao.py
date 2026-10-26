#!/usr/bin/env python3
"""Frozen-model ZSL leave-one-attribute-out: activity Top-1 and Macro-F1."""
from __future__ import annotations

import argparse
import gc
import hashlib
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

from evaluate_initial9_seen_predictability import (
    DATASETS, NAMES, attribute_names, dataset_inputs, read_json, sha256, validate_output, write_json,
)
from zshar.data import load_dataset_table
from zshar.evaluation.attribute_ablation import candidate_predictions, ordinal_distance_components, ordinal_scores
from zshar.evaluation.metrics import classification_summary
from zshar.evaluation.scoring import validation_attribute_weights
from zshar.semantics.prototypes import rank_activity_prototypes
from zshar.splits import load_sample_manifest, sample_evaluation_rows

OUTPUT = Path("/data/wanzhen/experiments/attribute_ablation/initial9_zsl_loao_seed42")
LEVELS = ["Low", "Medium", "High"]
LABELS = ["Intensity", "Cadence", "Smoothness", "Impulsiveness", "Inclination", "Posture variability",
          "Rotational motion", "Translational motion", "Motion regularity"]
CHINESE = ["运动强度", "动作频率", "平滑性", "冲击性", "姿态倾斜度", "姿态变化幅度", "旋转运动", "平移运动", "运动规律性"]


def source_inputs(ds):
    spec, run, splits, split_file = dataset_inputs(ds)
    config = read_json(run / "config.json")
    if config["prototype_metric"] != "ordinal_probability" or config["attribute_weighting"] != "validation_accuracy":
        raise ValueError("Unexpected original matching protocol")
    sample_file = PROJECT / "split/zero_shot_split" / Path(config["sample_manifest"]).name
    samples = load_sample_manifest(sample_file)
    if set(samples) != {s.index for s in splits}:
        raise ValueError("Sample/split manifest mismatch")
    for split in splits:
        metric = read_json(run / f"split_{split.index:03d}/metrics.json")
        if metric["attribute_vectors"]["sha256"] != sha256(spec.attributes_path):
            raise ValueError(f"Original test prototypes differ from initial v1.5: {ds}")
    return spec, run, splits, samples, sample_file, split_file


def test_probabilities(directory, model_dir, frame, feature_columns, signature):
    """Load a validated probability cache or perform one frozen-model inference."""
    metadata_path = directory / "probability_manifest.json"
    cache_path = directory / "test_probabilities.npz"
    if metadata_path.exists() and cache_path.exists():
        metadata = read_json(metadata_path)
        if metadata["signature"] != signature or metadata["npz_sha256"] != sha256(cache_path):
            raise ValueError(f"Probability cache provenance changed: {directory}")
        with np.load(cache_path) as payload:
            probability = payload["probabilities"]
        return probability, metadata["native_class_orders"]
    from autogluon.tabular import TabularPredictor
    manifest = read_json(model_dir / "attribute_backend_manifest.json")
    attributes = attribute_names()
    probability = np.zeros((len(frame), len(attributes), 3), dtype=np.float64)
    orders = []
    for j, attr in enumerate(attributes):
        entry = manifest["models"][attr]
        location = model_dir / entry["path"]
        if entry["backend"] == "constant":
            level = read_json(location / "constant_model.json")["constant_value"]
            order = [LEVELS.index(level)]
            probability[:, j, order[0]] = 1.
        else:
            predictor = TabularPredictor.load(str(location), verbosity=0)
            metadata = read_json(location / "backend_metadata.json")
            if metadata["feature_names"] != feature_columns or predictor.model_best != entry["metadata"]["predictor_model_best"]:
                raise ValueError("Saved predictor feature schema or selected model changed")
            native = predictor.predict_proba(frame[feature_columns])
            if not native.index.equals(frame.index) or native.columns.duplicated().any():
                raise ValueError("Prediction rows or class columns are misaligned")
            order = [LEVELS.index(str(label)) for label in native.columns]
            probability[:, j, order] = native.to_numpy(dtype=float)
            del native, predictor
            gc.collect()
        orders.append(order)
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, probabilities=probability)
    temporary.replace(cache_path)
    write_json(metadata_path, {"signature": signature, "native_class_orders": orders,
                              "npz_sha256": sha256(cache_path), "shape": list(probability.shape)})
    exported = frame[["window_index", "user_id", "activity_id", "activity_name"] + attributes].copy()
    for j, attr in enumerate(attributes):
        for level, name in enumerate(LEVELS):
            exported[f"{attr}__p_{name}"] = probability[:, j, level]
    exported.to_csv(directory / "test_probabilities.csv.gz", index=False, compression="gzip")
    return probability, orders


def evaluate_split(ds, spec, table, run, split, sample_ids, output, resume):
    directory = output / ds / f"split_{split.index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    model_dir = run / f"split_{split.index:03d}/models"
    frame = sample_evaluation_rows(table.frame, split, scenario="zsl",
        samples_per_activity=spec.eval_samples_per_activity, seed=42, manifest_ids=sample_ids)
    attributes = attribute_names()
    if set(frame.window_index) != sample_ids or not frame.activity_name.value_counts().eq(spec.eval_samples_per_activity).all():
        raise ValueError("Fixed ZSL test windows do not match the manifest")
    if len(frame) != len(split.unseen_activities)*spec.eval_samples_per_activity:
        raise ValueError("Unexpected number of fixed Unseen test windows")
    manifest_file = model_dir / "attribute_backend_manifest.json"
    signature = {"dataset": ds, "split_index": int(split.index),
        "model_manifest_sha256": sha256(manifest_file), "vectors_sha256": sha256(spec.attributes_path),
        "test_window_indices": list(map(int, frame.window_index)),
        "test_features_sha256": hashlib.sha256(np.ascontiguousarray(frame[table.feature_columns].to_numpy()).tobytes()).hexdigest()}
    result_path = directory / "metrics.json"
    if result_path.is_file() and resume:
        saved = read_json(result_path)
        if saved["signature"] != signature:
            raise ValueError("Completed split source inputs have changed")
        print(f"RESUME {ds} split={split.index}", flush=True)
        return saved["rows"]
    if result_path.exists():
        raise FileExistsError(f"Pass --resume for {directory}")
    manifest = read_json(manifest_file)
    weights = validation_attribute_weights(attributes, manifest["validation_report"])
    weight_values = [weights[a] for a in attributes]
    candidates = list(split.unseen_activities)
    prototypes = np.array([[LEVELS.index(table.activity_vectors[c][a]) for a in attributes] for c in candidates])
    probability, orders = test_probabilities(directory, model_dir, frame, table.feature_columns, signature)
    components = ordinal_distance_components(probability, prototypes, orders)
    baseline_scores = ordinal_scores(components, weight_values)
    baseline_prediction = candidate_predictions(baseline_scores, candidates)
    original_directory = run / f"split_{split.index:03d}"
    original = pd.read_csv(original_directory / "details.csv").set_index("window_index")
    if set(original.index) != set(frame.window_index):
        raise ValueError("Historical test IDs mismatch")
    original_prediction = original.loc[frame.window_index, "predicted_activity"].to_numpy(dtype=str)
    if not np.array_equal(original_prediction, baseline_prediction):
        mismatch = int(np.sum(original_prediction != baseline_prediction))
        raise ValueError(f"Full 9D baseline does not reproduce original predictions: {ds}/{split.index}, {mismatch} mismatches")
    baseline = classification_summary(frame.activity_name, baseline_prediction)
    original_metric = read_json(original_directory / "metrics.json")["activity"]
    for key in ("accuracy", "macro_f1"):
        if not np.isclose(baseline[key], original_metric[key], atol=1e-12, rtol=0):
            raise ValueError("Full 9D metrics differ from historical metrics")
    # Cross-check vectorized matching with the original reference implementation.
    max_reference_error = 0.
    for removed in [None] + list(range(9)):
        selected = [a for j,a in enumerate(attributes) if j != removed]
        scores = ordinal_scores(components, weight_values, removed)
        for i in range(min(3, len(frame))):
            distributions = {}
            for j, attr in enumerate(attributes):
                if j == removed:
                    continue
                native = {LEVELS[k]: max(0., float(probability[i,j,k])) for k in orders[j]}
                total = sum(native.values())
                distributions[attr] = {key:value/total for key,value in native.items()}
            reference = rank_activity_prototypes(predicted_values={},
                activity_vectors={c:table.activity_vectors[c] for c in candidates}, attribute_cols=selected,
                metric="ordinal_probability", attribute_weights=weights,
                probability_distributions=distributions)
            by_class = {c:m.score for c,m in reference}
            error = max(abs(scores[i,k]-by_class[c]) for k,c in enumerate(candidates))
            max_reference_error = max(max_reference_error, float(error))
            if error > 1e-14:
                raise ValueError("Vectorized and reference ordinal matching differ")
    predictions = frame[["window_index", "user_id", "activity_id", "activity_name"]].copy()
    predictions["full9_prediction"] = baseline_prediction
    rows = []
    variants = [("full9", None)] + [(a,j) for j,a in enumerate(attributes)]
    effective_weights = {}
    for variant, removed in variants:
        scores = ordinal_scores(components, weight_values, removed)
        pred = candidate_predictions(scores, candidates)
        metric = classification_summary(frame.activity_name, pred)
        remaining = [a for j,a in enumerate(attributes) if j != removed]
        denominator = sum(weights[a] for a in remaining)
        effective_weights[variant] = {a:weights[a]/denominator for a in remaining}
        predictions[variant+"__prediction"] = pred
        predictions[variant+"__correct"] = pred == frame.activity_name.to_numpy()
        rows.append(dict(dataset=ds, split_index=int(split.index), variant=variant,
            removed_attribute="" if removed is None else variant, attribute_count=len(remaining),
            samples=len(frame), candidate_count=len(candidates),
            top1_accuracy=metric["accuracy"], macro_f1=metric["macro_f1"],
            baseline_top1_accuracy=baseline["accuracy"], baseline_macro_f1=baseline["macro_f1"],
            delta_top1=baseline["accuracy"]-metric["accuracy"],
            delta_macro_f1=baseline["macro_f1"]-metric["macro_f1"]))
    predictions.to_csv(directory / "activity_predictions.csv.gz", index=False, compression="gzip")
    write_json(directory / "matching.json", {"candidate_order":candidates,"attributes":attributes,
        "raw_seen_validation_weights":weights,"renormalized_weights":effective_weights,
        "candidate_prototypes":{c:table.activity_vectors[c] for c in candidates},
        "reference_score_max_abs_error":max_reference_error,
        "full9_historical_prediction_matches":len(frame)})
    write_json(result_path, {"signature":signature,"rows":rows})
    print(f"DONE {ds} split={split.index} baseline_top1={baseline['accuracy']:.5f} "
          f"baseline_f1={baseline['macro_f1']:.5f}", flush=True)
    return rows


def bar_panel(ax, means, deviations, title, limit, labels=False):
    y = np.arange(9)
    colors = ["#6c9fd1" if value >= 0 else "#d88663" for value in means]
    ax.barh(y, means, xerr=deviations, color=colors, height=.6,
            error_kw={"ecolor":"#444444","elinewidth":.9,"capsize":2.3})
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlim(-limit,limit)
    ax.set_yticks(y, LABELS if labels else [""]*9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="x",color="#e5e7eb",linewidth=.6)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0,labelsize=10)
    ax.tick_params(axis="x",labelsize=9)
    for i,value in enumerate(means):
        offset=limit*.025
        ax.text(value+(offset if value>=0 else -offset),i,f"{value:+.2f}",
                ha="left" if value>=0 else "right",va="center",fontsize=8,
                bbox={"facecolor":"white","edgecolor":"none","alpha":.8,"pad":.3})


def plot_results(output, summary, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    attrs=attribute_names()
    with plt.rc_context({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none"}):
        for metric,caption in [("top1","Top-1 accuracy"),("macro_f1","Activity Macro-F1")]:
            mean=summary.pivot(index="attribute",columns="dataset",values=f"{metric}_delta_mean_pp").reindex(index=attrs,columns=datasets)
            std=summary.pivot(index="attribute",columns="dataset",values=f"{metric}_delta_std_pp").reindex(index=attrs,columns=datasets)
            limit=max(1.,float(np.max(np.abs(mean.to_numpy())+std.to_numpy()))*1.22)
            fig,axes=plt.subplots(1,len(datasets),figsize=(4.1*len(datasets)+2.3,6.5),layout="constrained",squeeze=False)
            for j,ds in enumerate(datasets):
                bar_panel(axes[0,j],mean[ds].to_numpy(),std[ds].to_numpy(),NAMES[ds],limit,j==0)
                axes[0,j].set_xlabel("Delta (percentage points)",fontsize=9)
            fig.suptitle(f"Leave-one-attribute-out impact: {caption}\nFull 9D minus removed attribute | mean +/- SD over 20 fixed ZSL splits",fontsize=14)
            fig.supxlabel("Positive: removing the attribute hurts.   Negative: removing it improves performance.   Error bars: split SD, not confidence intervals.",fontsize=10)
            for extension in ("png","pdf","svg"):
                fig.savefig(output/f"loao_{metric}_impact_{len(datasets)}datasets.{extension}",dpi=300)
            plt.close(fig)
        for ds in datasets:
            data=summary[summary.dataset.eq(ds)].set_index("attribute").reindex(attrs)
            fig,axes=plt.subplots(1,2,figsize=(13,6.5),layout="constrained")
            for j,(metric,caption) in enumerate([("top1","Top-1 accuracy"),("macro_f1","Activity Macro-F1")]):
                mean=data[f"{metric}_delta_mean_pp"].to_numpy();std=data[f"{metric}_delta_std_pp"].to_numpy()
                limit=max(1.,float(np.max(np.abs(mean)+std))*1.22)
                bar_panel(axes[j],mean,std,caption,limit,j==0)
                axes[j].set_xlabel("Full 9D - ablated (percentage points)")
            fig.suptitle(f"{NAMES[ds]}: leave-one-attribute-out | mean +/- SD over 20 ZSL splits",fontsize=14)
            for extension in ("png","pdf","svg"):
                fig.savefig(output/ds/f"loao_impact.{extension}",dpi=300)
            plt.close(fig)


def summarize(output, rows, datasets):
    frame=pd.DataFrame(rows)
    if frame.duplicated(["dataset","split_index","variant"]).any():
        raise ValueError("Duplicate metrics")
    frame.to_csv(output/"per_split_metrics.csv",index=False)
    attrs=attribute_names()
    records=[]
    for ds in datasets:
        for attr in attrs:
            part=frame[frame.dataset.eq(ds)&frame.variant.eq(attr)]
            records.append(dict(dataset=ds,attribute=attr,split_count=len(part),
                top1_full_mean=part.baseline_top1_accuracy.mean(),top1_ablated_mean=part.top1_accuracy.mean(),
                top1_delta_mean_pp=100*part.delta_top1.mean(),top1_delta_std_pp=100*part.delta_top1.std(ddof=1),
                macro_f1_full_mean=part.baseline_macro_f1.mean(),macro_f1_ablated_mean=part.macro_f1.mean(),
                macro_f1_delta_mean_pp=100*part.delta_macro_f1.mean(),macro_f1_delta_std_pp=100*part.delta_macro_f1.std(ddof=1),
                top1_positive_splits=int((part.delta_top1>1e-12).sum()),
                top1_negative_splits=int((part.delta_top1 < -1e-12).sum()),
                macro_f1_positive_splits=int((part.delta_macro_f1>1e-12).sum()),
                macro_f1_negative_splits=int((part.delta_macro_f1 < -1e-12).sum())))
    summary=pd.DataFrame(records)
    summary.to_csv(output/"ablation_summary.csv",index=False)
    matrices={}
    for metric,field in [("top1","top1_accuracy"),("macro_f1","macro_f1")]:
        absolute=frame.groupby(["variant","dataset"])[field].mean().unstack().reindex(index=["full9"]+attrs,columns=datasets)*100
        absolute.to_csv(output/f"{metric}_full_and_ablated_percent.csv",float_format="%.10f")
        matrices[metric+"_absolute_pct"]=absolute
        for statistic in ("mean","std"):
            matrix=summary.pivot(index="attribute",columns="dataset",values=f"{metric}_delta_{statistic}_pp").reindex(index=attrs,columns=datasets)
            matrix.to_csv(output/f"{metric}_delta_{statistic}_pp_9x{len(datasets)}.csv",float_format="%.10f")
            matrices[metric+"_delta_"+statistic+"_pp"]=matrix
    baseline=frame[frame.variant.eq("full9")].groupby("dataset")[["top1_accuracy","macro_f1"]].agg(["mean","std"]).reindex(datasets)
    baseline.to_csv(output/"full9_baseline.csv")
    extras=Path("/data/wanzhen/experiments/attribute_predictability/initial9_zsl_seen_validation_seed42/cache/python_deps")
    if extras.exists():sys.path.insert(0,str(extras))
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        pass
    else:
        with pd.ExcelWriter(output/"initial9_zsl_loao_results.xlsx") as writer:
            for name,matrix in matrices.items():matrix.rename(columns=NAMES).to_excel(writer,sheet_name=name[:31])
            summary.to_excel(writer,sheet_name="summary",index=False)
            frame.to_excel(writer,sheet_name="per_split",index=False)
            baseline.to_excel(writer,sheet_name="full9_baseline")
    plot_results(output,summary,datasets)
    def table(matrix):
        lines=["| 移除属性 | "+" | ".join(NAMES[d] for d in datasets)+" |","|---|"+"---:|"*len(datasets)]
        for attr,name in zip(attrs,CHINESE):
            lines.append("| "+name+" | "+" | ".join(f"{v:+.3f}" for v in matrix.loc[attr])+" |")
        return "\n".join(lines)
    report=["# 初始9维属性的逐属性移除实验\n",
        "固定 initial v1.5、独立 AutoGluon 模型、seed=42 和各数据集原有 20 个 ZSL split。候选类别仅为该 split 的 Unseen 类。",
        "PAMAP2 每个 Unseen 类固定50个窗口，其余数据集固定30个窗口；不使用 Seen 验证样本替代 ZSL 测试样本。\n",
        "## 完整9维基线\n","| 数据集 | Top-1 (%) | 活动 Macro-F1 (%) |","|---|---:|---:|"]
    for ds in datasets:
        base=frame[frame.dataset.eq(ds)&frame.variant.eq("full9")]
        report.append(f"| {NAMES[ds]} | {base.top1_accuracy.mean()*100:.3f} | {base.macro_f1.mean()*100:.3f} |")
    report += ["\n## Top-1 差值（百分点）\n",table(matrices['top1_delta_mean_pp']),
        "\n## 活动 Macro-F1 差值（百分点）\n",table(matrices['macro_f1_delta_mean_pp']),
        "\n## 口径与解释\n",
        "- 差值 = 完整9维指标 − 移除该维后的指标。正值表示移除后下降，负值表示移除后改善，零值表示当前设置下没有净变化。",
        "- 在同一个 split、同一批窗口、同一候选类别和固定的概率输出上作配对比较，再对20个 split 等权求均值。",
        "- 权重沿用原模型 Seen 外层验证准确率；移除时删除该属性概率/原型坐标与权重项，再以剩余权重之和重新归一化。不重新训练、重新选模、重新校准或根据测试标签调整权重。",
        "- 序数概率距离为各等级概率加权的绝对距离，Low/Medium/High 对应0/0.5/1；保留原候选顺序和 np.argsort 的并列处理方式。",
        "- Macro-F1 是最终活动分类的 Macro-F1，不是属性预测 Macro-F1。所有固定测试类均纳入评价。",
        "- 图中误差线为20个配对 split 差值的样本标准差（ddof=1），不是标准误或置信区间；这些 split 共享样本，不能视为独立重复实验。",
        "- 每个完整9维配置均须与历史活动预测逐样本完全一致，并复核Top-1和Macro-F1。",
        "- 实验评估冻结系统中的条件边际贡献；它不能保证每个属性都必要。零/负差值应如实保留，不能据测试结果删除属性后再把结果作为独立测试结论。\n",
        "完整值与删除后绝对值见 `top1_full_and_ablated_percent.csv`、`macro_f1_full_and_ablated_percent.csv`。",
        "Excel 包含两项指标的完整值、差值、差值标准差和逐split明细。PNG/PDF/SVG 横向条形图分别展示两项指标。",
        "每个split保存测试属性概率、十种配置的活动预测、初始候选原型、原始/重新归一化权重和输入哈希。"]
    (output/"REPORT_zh.md").write_text("\n".join(report)+"\n")
    write_json(output/"status.json",{"complete":len(frame)==len(datasets)*200,
        "completed_splits":{ds:int(frame[frame.dataset.eq(ds)].split_index.nunique()) for ds in datasets},
        "variant_split_rows":len(frame)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    output=validate_output(args.output)
    inputs={ds:source_inputs(ds) for ds in args.datasets}
    protocol={"experiment":"initial9_v1.5_frozen_output_ZSL_leave_one_attribute_out","seed":42,
        "delta":"full9 - removed_attribute","error_bars":"sample SD of 20 paired class-split differences (ddof=1)",
        "metric":"ordinal_probability","weighting":"original Seen validation accuracy; renormalize remaining weights",
        "no_training_or_model_selection":True,"unseen_test_used_only_for_evaluation":True,
        "datasets":{ds:{"run":str(data[1]),"vectors":str(data[0].attributes_path),
            "vectors_sha256":sha256(data[0].attributes_path),"samples_sha256":sha256(data[4]),
            "splits_sha256":sha256(data[5]),"splits":20} for ds,data in inputs.items()}}
    write_json(output/"protocol.json",protocol)
    if args.check:print(json.dumps(protocol,indent=2));return
    if args.summarize_only:
        rows=[]
        for ds in args.datasets:
            files=sorted((output/ds).glob('split_*/metrics.json'))
            if len(files)!=20:raise ValueError("Summary requires all twenty splits")
            for p in files:rows.extend(read_json(p)["rows"])
        summarize(output,rows,args.datasets);return
    import torch
    torch.set_num_threads(4)
    rows=[]
    for ds,(spec,run,splits,samples,_,_) in inputs.items():
        table_data=load_dataset_table(spec)
        if table_data.attribute_columns!=attribute_names():raise ValueError("Unexpected attribute order")
        for split in splits:
            started=time.monotonic()
            rows.extend(evaluate_split(ds,spec,table_data,run,split,samples[split.index],output,args.resume))
            print(f"ELAPSED {ds} split={split.index} seconds={time.monotonic()-started:.1f}",flush=True)
        del table_data;gc.collect()
    summarize(output,rows,args.datasets)
    print("FINISHED "+str(output),flush=True)


if __name__=="__main__":main()
