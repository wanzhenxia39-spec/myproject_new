# USC-HAD 按活动统计的 LOAO 结果入口

差值 = 完整9维 − 删除后8维（pp）。主分析第二面板为活动类别F1；另提供一对其余二分类Macro-F1，两者口径不同。

- [Top-1＋类别F1：属性×活动总览](activity_attribute_impact_top1_f1.png) · [PDF](activity_attribute_impact_top1_f1.pdf)
- [Top-1＋一对其余Macro-F1：总览](activity_attribute_impact_top1_ovr_macro_f1.png) · [PDF](activity_attribute_impact_top1_ovr_macro_f1.pdf)
- [Excel完整对照表](activity_loao_results.xlsx)
- [中文指标说明与全部差值矩阵](REPORT_zh.md)

|活动|作为Unseen的split数|Top-1＋类别F1|Top-1＋一对其余Macro-F1|
|---|---:|---|---|
|Walking Forward|12/20|[PNG](per_activity/01_Walking_Forward_impact.png) · [PDF](per_activity/01_Walking_Forward_impact.pdf)|[PNG](per_activity_ovr_macro_f1/01_Walking_Forward_impact.png) · [PDF](per_activity_ovr_macro_f1/01_Walking_Forward_impact.pdf)|
|Walking Left|6/20|[PNG](per_activity/02_Walking_Left_impact.png) · [PDF](per_activity/02_Walking_Left_impact.pdf)|[PNG](per_activity_ovr_macro_f1/02_Walking_Left_impact.png) · [PDF](per_activity_ovr_macro_f1/02_Walking_Left_impact.pdf)|
|Walking Right|7/20|[PNG](per_activity/03_Walking_Right_impact.png) · [PDF](per_activity/03_Walking_Right_impact.pdf)|[PNG](per_activity_ovr_macro_f1/03_Walking_Right_impact.png) · [PDF](per_activity_ovr_macro_f1/03_Walking_Right_impact.pdf)|
|Walking Upstairs|9/20|[PNG](per_activity/04_Walking_Upstairs_impact.png) · [PDF](per_activity/04_Walking_Upstairs_impact.pdf)|[PNG](per_activity_ovr_macro_f1/04_Walking_Upstairs_impact.png) · [PDF](per_activity_ovr_macro_f1/04_Walking_Upstairs_impact.pdf)|
|Walking Downstairs|8/20|[PNG](per_activity/05_Walking_Downstairs_impact.png) · [PDF](per_activity/05_Walking_Downstairs_impact.pdf)|[PNG](per_activity_ovr_macro_f1/05_Walking_Downstairs_impact.png) · [PDF](per_activity_ovr_macro_f1/05_Walking_Downstairs_impact.pdf)|
|Running Forward|4/20|[PNG](per_activity/06_Running_Forward_impact.png) · [PDF](per_activity/06_Running_Forward_impact.pdf)|[PNG](per_activity_ovr_macro_f1/06_Running_Forward_impact.png) · [PDF](per_activity_ovr_macro_f1/06_Running_Forward_impact.pdf)|
|Jumping Up|5/20|[PNG](per_activity/07_Jumping_Up_impact.png) · [PDF](per_activity/07_Jumping_Up_impact.pdf)|[PNG](per_activity_ovr_macro_f1/07_Jumping_Up_impact.png) · [PDF](per_activity_ovr_macro_f1/07_Jumping_Up_impact.pdf)|
|Sitting|4/20|[PNG](per_activity/08_Sitting_impact.png) · [PDF](per_activity/08_Sitting_impact.pdf)|[PNG](per_activity_ovr_macro_f1/08_Sitting_impact.png) · [PDF](per_activity_ovr_macro_f1/08_Sitting_impact.pdf)|
|Standing|5/20|[PNG](per_activity/09_Standing_impact.png) · [PDF](per_activity/09_Standing_impact.pdf)|[PNG](per_activity_ovr_macro_f1/09_Standing_impact.png) · [PDF](per_activity_ovr_macro_f1/09_Standing_impact.pdf)|
|Sleeping|6/20|[PNG](per_activity/10_Sleeping_impact.png) · [PDF](per_activity/10_Sleeping_impact.pdf)|[PNG](per_activity_ovr_macro_f1/10_Sleeping_impact.png) · [PDF](per_activity_ovr_macro_f1/10_Sleeping_impact.pdf)|
|Elevator Up|6/20|[PNG](per_activity/11_Elevator_Up_impact.png) · [PDF](per_activity/11_Elevator_Up_impact.pdf)|[PNG](per_activity_ovr_macro_f1/11_Elevator_Up_impact.png) · [PDF](per_activity_ovr_macro_f1/11_Elevator_Up_impact.pdf)|
|Elevator Down|8/20|[PNG](per_activity/12_Elevator_Down_impact.png) · [PDF](per_activity/12_Elevator_Down_impact.pdf)|[PNG](per_activity_ovr_macro_f1/12_Elevator_Down_impact.png) · [PDF](per_activity_ovr_macro_f1/12_Elevator_Down_impact.pdf)|

代码入口：`/home/wanzhen/attribute_prototype_analysis/analyze_initial9_loao_by_activity.py`。
复现命令：

```bash
bash /home/wanzhen/attribute_prototype_analysis/run_initial9_loao_by_activity.sh uschad
```
