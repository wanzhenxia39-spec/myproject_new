# A2 v4 LOAO 结果索引

推荐先看主图：四个全局公共属性，在五个数据集各20个 split 上的配对差值。完整基线仍使用各 split 自己的完整细化属性集。

- [Top-1 主图](loao_top1_common_impact_5datasets.png) · [PDF](loao_top1_common_impact_5datasets.pdf)
- [活动 Macro-F1 主图](loao_macro_f1_common_impact_5datasets.png) · [PDF](loao_macro_f1_common_impact_5datasets.pdf)
- [Excel 结果表](refined_v4_loao_results.xlsx)
- [中文分析与协议说明](REPORT_zh.md)

## 补充结果

- dataset_common：每个数据集各自20个 split 均存在的属性，优先于稀疏条件统计。
- exact_semantics：按名称、定义和等级描述精确分组的全部属性图；标签 s 为对应 split，n=1 无标准差。
- all_attributes：同名属性合并的探索性索引，可能混合不同语义定义，不能作为必要性的主证据。
- exact_semantics_catalog.csv：每个精确定义组的 split 清单、定义、指标差值与标准差。
- per_split_metrics.csv：所有1105个完整/移除配置的绝对值和配对差值。
- input_audit.json：最新版原型、生成计划及对应模型运行的哈希；各配置保存实际重载预测记录、概率和活动预测。
- verification.json / artifact_verification.json：配置计数及对保存概率、预测和指标的复核。

差值 = 完整 − 删除，单位 pp；误差线为20个或 n 个配对差值的标准差，不是置信区间。
