# MHEALTH 按活动统计的 LOAO 结果入口

差值 = 完整9维 − 删除后8维（pp）。主分析第二面板为活动类别F1；另提供一对其余二分类Macro-F1，两者口径不同。

- [Top-1＋类别F1：属性×活动总览](activity_attribute_impact_top1_f1.png) · [PDF](activity_attribute_impact_top1_f1.pdf)
- [Top-1＋一对其余Macro-F1：总览](activity_attribute_impact_top1_ovr_macro_f1.png) · [PDF](activity_attribute_impact_top1_ovr_macro_f1.pdf)
- [Excel完整对照表](activity_loao_results.xlsx)
- [中文指标说明与全部差值矩阵](REPORT_zh.md)

|活动|作为Unseen的split数|Top-1＋类别F1|Top-1＋一对其余Macro-F1|
|---|---:|---|---|
|standing still|12/20|[PNG](per_activity/01_standing_still_impact.png) · [PDF](per_activity/01_standing_still_impact.pdf)|[PNG](per_activity_ovr_macro_f1/01_standing_still_impact.png) · [PDF](per_activity_ovr_macro_f1/01_standing_still_impact.pdf)|
|sitting and relaxing|6/20|[PNG](per_activity/02_sitting_and_relaxing_impact.png) · [PDF](per_activity/02_sitting_and_relaxing_impact.pdf)|[PNG](per_activity_ovr_macro_f1/02_sitting_and_relaxing_impact.png) · [PDF](per_activity_ovr_macro_f1/02_sitting_and_relaxing_impact.pdf)|
|lying down|7/20|[PNG](per_activity/03_lying_down_impact.png) · [PDF](per_activity/03_lying_down_impact.pdf)|[PNG](per_activity_ovr_macro_f1/03_lying_down_impact.png) · [PDF](per_activity_ovr_macro_f1/03_lying_down_impact.pdf)|
|walking|9/20|[PNG](per_activity/04_walking_impact.png) · [PDF](per_activity/04_walking_impact.pdf)|[PNG](per_activity_ovr_macro_f1/04_walking_impact.png) · [PDF](per_activity_ovr_macro_f1/04_walking_impact.pdf)|
|climbing stairs|8/20|[PNG](per_activity/05_climbing_stairs_impact.png) · [PDF](per_activity/05_climbing_stairs_impact.pdf)|[PNG](per_activity_ovr_macro_f1/05_climbing_stairs_impact.png) · [PDF](per_activity_ovr_macro_f1/05_climbing_stairs_impact.pdf)|
|waist bends forward|4/20|[PNG](per_activity/06_waist_bends_forward_impact.png) · [PDF](per_activity/06_waist_bends_forward_impact.pdf)|[PNG](per_activity_ovr_macro_f1/06_waist_bends_forward_impact.png) · [PDF](per_activity_ovr_macro_f1/06_waist_bends_forward_impact.pdf)|
|frontal elevation of arms|5/20|[PNG](per_activity/07_frontal_elevation_of_arms_impact.png) · [PDF](per_activity/07_frontal_elevation_of_arms_impact.pdf)|[PNG](per_activity_ovr_macro_f1/07_frontal_elevation_of_arms_impact.png) · [PDF](per_activity_ovr_macro_f1/07_frontal_elevation_of_arms_impact.pdf)|
|knees bending (crouching)|4/20|[PNG](per_activity/08_knees_bending__crouching_impact.png) · [PDF](per_activity/08_knees_bending__crouching_impact.pdf)|[PNG](per_activity_ovr_macro_f1/08_knees_bending__crouching_impact.png) · [PDF](per_activity_ovr_macro_f1/08_knees_bending__crouching_impact.pdf)|
|cycling|5/20|[PNG](per_activity/09_cycling_impact.png) · [PDF](per_activity/09_cycling_impact.pdf)|[PNG](per_activity_ovr_macro_f1/09_cycling_impact.png) · [PDF](per_activity_ovr_macro_f1/09_cycling_impact.pdf)|
|jogging|6/20|[PNG](per_activity/10_jogging_impact.png) · [PDF](per_activity/10_jogging_impact.pdf)|[PNG](per_activity_ovr_macro_f1/10_jogging_impact.png) · [PDF](per_activity_ovr_macro_f1/10_jogging_impact.pdf)|
|running|6/20|[PNG](per_activity/11_running_impact.png) · [PDF](per_activity/11_running_impact.pdf)|[PNG](per_activity_ovr_macro_f1/11_running_impact.png) · [PDF](per_activity_ovr_macro_f1/11_running_impact.pdf)|
|jump front & back|8/20|[PNG](per_activity/12_jump_front___back_impact.png) · [PDF](per_activity/12_jump_front___back_impact.pdf)|[PNG](per_activity_ovr_macro_f1/12_jump_front___back_impact.png) · [PDF](per_activity_ovr_macro_f1/12_jump_front___back_impact.pdf)|

代码入口：`/home/wanzhen/attribute_prototype_analysis/analyze_initial9_loao_by_activity.py`。
复现命令：

```bash
bash /home/wanzhen/attribute_prototype_analysis/run_initial9_loao_by_activity.sh mhealth
```
