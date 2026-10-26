# 初始9维 ZSL 逐属性移除：每个配置独立重新预测

本次按用户明确指定，全部结果、日志、预测概率、缓存和测试临时文件保存在本系统盘目录：

`/home/wanzhen/attribute_prototype_analysis/new_results/initial9_zsl_loao_new/`

入口代码：`/home/wanzhen/attribute_prototype_analysis/reinfer_initial9_zsl_loao.py`。

## 实际执行方式

- 五个数据集各20个 seed=42 ZSL split；与旧实验使用相同的模型、initial v1.5 原型、固定测试窗口、候选类别顺序和原始 Seen 验证权重。
- 每个 split 的完整9维基线独立重新加载9个属性分类器，并重新调用各自的 `predict_proba`。
- 九个移除配置各自重新加载保留的8个属性分类器并重新预测，不复用其他配置的模型对象或概率。被移除属性的分类器不调用。
- 仅用本配置新预测的8维概率与8维候选原型进行序数距离匹配，以剩余权重之和重新归一化。
- 常量属性通过重新加载其已保存的 ConstantAttributeBackend 并调用 `predict_proba` 处理；不将其伪称为 AutoGluon 训练模型。
- 不重新训练、选模、调参或重新估计权重。

合计100个 split、1,000个配置、8,100次属性模型加载和预测调用。其中预期7,920次为 AutoGluon 属性预测器，180次为常量预测器。

每个配置单独保存：

- `inference_calls.json`：实际加载路径、属性、后端、调用次数、样本数、概率哈希和耗时。
- `fresh_probabilities.npz`：该配置新产生的概率，完整基线为9维，移除配置严格为8维。
- `fresh_test_predictions.csv.gz`：活动标签、保留属性目标等级、三级概率和活动预测。
- `metrics.json`：本配置的指标、有效权重及与旧结果的比较。

旧目录 `../initial9_zsl_loao/` 的概率和活动预测只用于事后比较，不参与本次新匹配。

## 汇总与图形

- `initial9_zsl_loao_results.xlsx`：Top-1及活动Macro-F1的完整值、删除后值、差值和标准差。
- `top1_delta_mean_pp_9x5.csv`、`macro_f1_delta_mean_pp_9x5.csv`：完整9维减删除后的配对差值均值，单位百分点。
- `loao_top1_impact_5datasets.{png,pdf,svg}`、`loao_macro_f1_impact_5datasets.{png,pdf,svg}`：五数据集横向条形图；误差线为20个配对split差值的样本标准差。
- 各数据集下 `loao_impact.{png,pdf,svg}` 为单独两面板图。
- `new_vs_previous.csv`：逐配置概率变化、活动预测变化及两项指标变化。
- `reinference_checks.json`：全部新旧比较汇总和实际调用次数。

确定性推理下，重新加载相同模型、输入相同IMU特征通常会得到相同概率。本次以实际比较结果为准，保留零变化和负贡献，不以期望结果替代实测。

## 重跑命令

```bash
OUT=/home/wanzhen/attribute_prototype_analysis/new_results/initial9_zsl_loao_new
CUDA_VISIBLE_DEVICES='' \
PYTHONPYCACHEPREFIX="$OUT/cache/pycache" \
MPLCONFIGDIR="$OUT/cache/matplotlib" TMPDIR="$OUT/cache/tmp" \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/wanzhen/miniconda3/envs/zara/bin/python -u \
 /home/wanzhen/attribute_prototype_analysis/reinfer_initial9_zsl_loao.py \
 --workers 5 --threads 2 --resume > "$OUT/evaluation.log" 2>&1
```

`--resume` 仅跳过本次已经完成且输入哈希一致的配置；首次执行的配置均进行真实模型加载和预测。`--summarize-only` 只重生成图表。
