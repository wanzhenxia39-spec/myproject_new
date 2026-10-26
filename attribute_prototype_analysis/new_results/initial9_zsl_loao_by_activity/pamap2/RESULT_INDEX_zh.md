# PAMAP2 按活动统计的 LOAO 结果入口

差值 = 完整9维 − 删除后8维（pp）。主分析第二面板为活动类别F1；另提供一对其余二分类Macro-F1，两者口径不同。

- [Top-1＋类别F1：属性×活动总览](activity_attribute_impact_top1_f1.png) · [PDF](activity_attribute_impact_top1_f1.pdf)
- [Top-1＋一对其余Macro-F1：总览](activity_attribute_impact_top1_ovr_macro_f1.png) · [PDF](activity_attribute_impact_top1_ovr_macro_f1.pdf)
- [Excel完整对照表](activity_loao_results.xlsx)
- [中文指标说明与全部差值矩阵](REPORT_zh.md)

|活动|作为Unseen的split数|Top-1＋类别F1|Top-1＋一对其余Macro-F1|
|---|---:|---|---|
|lying|15/20|[PNG](per_activity/01_lying_impact.png) · [PDF](per_activity/01_lying_impact.pdf)|[PNG](per_activity_ovr_macro_f1/01_lying_impact.png) · [PDF](per_activity_ovr_macro_f1/01_lying_impact.pdf)|
|sitting|9/20|[PNG](per_activity/02_sitting_impact.png) · [PDF](per_activity/02_sitting_impact.pdf)|[PNG](per_activity_ovr_macro_f1/02_sitting_impact.png) · [PDF](per_activity_ovr_macro_f1/02_sitting_impact.pdf)|
|standing|9/20|[PNG](per_activity/03_standing_impact.png) · [PDF](per_activity/03_standing_impact.pdf)|[PNG](per_activity_ovr_macro_f1/03_standing_impact.png) · [PDF](per_activity_ovr_macro_f1/03_standing_impact.pdf)|
|walking|9/20|[PNG](per_activity/04_walking_impact.png) · [PDF](per_activity/04_walking_impact.pdf)|[PNG](per_activity_ovr_macro_f1/04_walking_impact.png) · [PDF](per_activity_ovr_macro_f1/04_walking_impact.pdf)|
|running|7/20|[PNG](per_activity/05_running_impact.png) · [PDF](per_activity/05_running_impact.pdf)|[PNG](per_activity_ovr_macro_f1/05_running_impact.png) · [PDF](per_activity_ovr_macro_f1/05_running_impact.pdf)|
|cycling|4/20|[PNG](per_activity/06_cycling_impact.png) · [PDF](per_activity/06_cycling_impact.pdf)|[PNG](per_activity_ovr_macro_f1/06_cycling_impact.png) · [PDF](per_activity_ovr_macro_f1/06_cycling_impact.pdf)|
|Nordic walking|4/20|[PNG](per_activity/07_Nordic_walking_impact.png) · [PDF](per_activity/07_Nordic_walking_impact.pdf)|[PNG](per_activity_ovr_macro_f1/07_Nordic_walking_impact.png) · [PDF](per_activity_ovr_macro_f1/07_Nordic_walking_impact.pdf)|
|ascending stairs|4/20|[PNG](per_activity/08_ascending_stairs_impact.png) · [PDF](per_activity/08_ascending_stairs_impact.pdf)|[PNG](per_activity_ovr_macro_f1/08_ascending_stairs_impact.png) · [PDF](per_activity_ovr_macro_f1/08_ascending_stairs_impact.pdf)|
|descending stairs|4/20|[PNG](per_activity/09_descending_stairs_impact.png) · [PDF](per_activity/09_descending_stairs_impact.pdf)|[PNG](per_activity_ovr_macro_f1/09_descending_stairs_impact.png) · [PDF](per_activity_ovr_macro_f1/09_descending_stairs_impact.pdf)|
|vacuum cleaning|4/20|[PNG](per_activity/10_vacuum_cleaning_impact.png) · [PDF](per_activity/10_vacuum_cleaning_impact.pdf)|[PNG](per_activity_ovr_macro_f1/10_vacuum_cleaning_impact.png) · [PDF](per_activity_ovr_macro_f1/10_vacuum_cleaning_impact.pdf)|
|ironing|4/20|[PNG](per_activity/11_ironing_impact.png) · [PDF](per_activity/11_ironing_impact.pdf)|[PNG](per_activity_ovr_macro_f1/11_ironing_impact.png) · [PDF](per_activity_ovr_macro_f1/11_ironing_impact.pdf)|
|rope jumping|7/20|[PNG](per_activity/12_rope_jumping_impact.png) · [PDF](per_activity/12_rope_jumping_impact.pdf)|[PNG](per_activity_ovr_macro_f1/12_rope_jumping_impact.png) · [PDF](per_activity_ovr_macro_f1/12_rope_jumping_impact.pdf)|

代码入口：`/home/wanzhen/attribute_prototype_analysis/analyze_initial9_loao_by_activity.py`。
复现命令：

```bash
bash /home/wanzhen/attribute_prototype_analysis/run_initial9_loao_by_activity.sh pamap2
```
