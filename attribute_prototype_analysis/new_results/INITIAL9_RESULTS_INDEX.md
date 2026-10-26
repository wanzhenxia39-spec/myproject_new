# 初始9维属性实验结果入口

两组结果已于 2026-10-23 按用户要求迁移到当前系统盘目录，所在文件系统为 `/dev/nvme0n1p4`。
所有原始结果文件均已逐文件核对 SHA256；原数据盘上的这两个结果目录已移除。

## 属性可预测性

目录：`initial9_seen_predictability/`

- [Excel：Macro-F1、EOD 两张9×5矩阵及标准差、基线、明细](initial9_seen_predictability/attribute_predictability_9x5.xlsx)
- [中文报告与统计说明](initial9_seen_predictability/REPORT_zh.md)
- [Macro-F1 热力图 PDF](initial9_seen_predictability/macro_f1_heatmap_9x5.pdf) · [PNG](initial9_seen_predictability/macro_f1_heatmap_9x5.png)
- [EOD 热力图 PDF](initial9_seen_predictability/eod_heatmap_9x5.pdf) · [PNG](initial9_seen_predictability/eod_heatmap_9x5.png)
- 各数据集各split的验证预测、三等级概率、划分与审计文件均保留在该子目录。

## ZSL逐属性移除实验

目录：`initial9_zsl_loao/`

- [Excel：完整9维、删除后值、配对差值和标准差](initial9_zsl_loao/initial9_zsl_loao_results.xlsx)
- [中文报告与两张9×5差值矩阵](initial9_zsl_loao/REPORT_zh.md)
- [Top-1 差值条形图 PDF](initial9_zsl_loao/loao_top1_impact_5datasets.pdf) · [PNG](initial9_zsl_loao/loao_top1_impact_5datasets.png)
- [活动 Macro-F1 差值条形图 PDF](initial9_zsl_loao/loao_macro_f1_impact_5datasets.pdf) · [PNG](initial9_zsl_loao/loao_macro_f1_impact_5datasets.png)
- 各数据集单独图、测试概率、十种配置的活动预测、原型、权重及逐split明细均保留。

## 迁移记录

两组原始目录共 10,007 个文件，逐文件复制校验通过；迁移清单位于 `_migration_manifests/`。
原始协议、日志、README 与代码快照保留生成时的路径作为历史记录。
当前结果的位置以本入口和各子目录的 `LOCATION.md` 为准。本次没有重新训练或重算指标。


## 每个删除配置独立重新预测的新实验

目录：`initial9_zsl_loao_new/`。完整9维和每个8维删除配置均独立重新加载已保存的保留属性分类器并重新调用 predict_proba。

- [新版 Excel：完整值、移除后值、配对差值和标准差](initial9_zsl_loao_new/initial9_zsl_loao_results.xlsx)
- [Top-1 条形图 PDF](initial9_zsl_loao_new/loao_top1_impact_5datasets.pdf) · [PNG](initial9_zsl_loao_new/loao_top1_impact_5datasets.png)
- [活动 Macro-F1 条形图 PDF](initial9_zsl_loao_new/loao_macro_f1_impact_5datasets.pdf) · [PNG](initial9_zsl_loao_new/loao_macro_f1_impact_5datasets.png)
- [新实验报告及与旧实验比较](initial9_zsl_loao_new/REPORT_zh.md)
- [实际加载/预测调用次数与新旧差异](initial9_zsl_loao_new/reinference_checks.json)


## MHEALTH 规律性重标注后的最新四数据集 LOAO

2026-10-24：本次完整 LOAO 结果已迁移至系统盘正式 results 目录，原始153个文件的 SHA256 校验通过。
MHEALTH 使用重标注后重新训练的模型；PAMAP2、USC-HAD、DSADS 沿用既有结果，不含 WISDM。

- 当前目录：`/home/wanzhen/attribute_prototype_analysis/results/mhealth_initial9_regularity_seed42/loao_4datasets`
- [中文报告](../results/mhealth_initial9_regularity_seed42/loao_4datasets/REPORT_zh.md)
- [Excel 完整结果](../results/mhealth_initial9_regularity_seed42/loao_4datasets/initial9_zsl_loao_results.xlsx)
- [迁移与位置说明](../results/mhealth_initial9_regularity_seed42/loao_4datasets/LOCATION.md)

此前 initial9_zsl_loao 和 initial9_zsl_loao_new 保留为各自历史实验。
