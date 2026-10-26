# WISDM 按活动统计的 LOAO 结果入口

差值 = 完整9维 − 删除后8维（pp）。主分析第二面板为活动类别F1；另提供一对其余二分类Macro-F1，两者口径不同。

- [Top-1＋类别F1：属性×活动总览](activity_attribute_impact_top1_f1.png) · [PDF](activity_attribute_impact_top1_f1.pdf)
- [Top-1＋一对其余Macro-F1：总览](activity_attribute_impact_top1_ovr_macro_f1.png) · [PDF](activity_attribute_impact_top1_ovr_macro_f1.pdf)
- [Excel完整对照表](activity_loao_results.xlsx)
- [中文指标说明与全部差值矩阵](REPORT_zh.md)

|活动|作为Unseen的split数|Top-1＋类别F1|Top-1＋一对其余Macro-F1|
|---|---:|---|---|
|walking|9/20|[PNG](per_activity/01_walking_impact.png) · [PDF](per_activity/01_walking_impact.pdf)|[PNG](per_activity_ovr_macro_f1/01_walking_impact.png) · [PDF](per_activity_ovr_macro_f1/01_walking_impact.pdf)|
|jogging|10/20|[PNG](per_activity/02_jogging_impact.png) · [PDF](per_activity/02_jogging_impact.pdf)|[PNG](per_activity_ovr_macro_f1/02_jogging_impact.png) · [PDF](per_activity_ovr_macro_f1/02_jogging_impact.pdf)|
|stairs|5/20|[PNG](per_activity/03_stairs_impact.png) · [PDF](per_activity/03_stairs_impact.pdf)|[PNG](per_activity_ovr_macro_f1/03_stairs_impact.png) · [PDF](per_activity_ovr_macro_f1/03_stairs_impact.pdf)|
|sitting|6/20|[PNG](per_activity/04_sitting_impact.png) · [PDF](per_activity/04_sitting_impact.pdf)|[PNG](per_activity_ovr_macro_f1/04_sitting_impact.png) · [PDF](per_activity_ovr_macro_f1/04_sitting_impact.pdf)|
|standing|10/20|[PNG](per_activity/05_standing_impact.png) · [PDF](per_activity/05_standing_impact.pdf)|[PNG](per_activity_ovr_macro_f1/05_standing_impact.png) · [PDF](per_activity_ovr_macro_f1/05_standing_impact.pdf)|
|typing|4/20|[PNG](per_activity/06_typing_impact.png) · [PDF](per_activity/06_typing_impact.pdf)|[PNG](per_activity_ovr_macro_f1/06_typing_impact.png) · [PDF](per_activity_ovr_macro_f1/06_typing_impact.pdf)|
|brushing teeth|7/20|[PNG](per_activity/07_brushing_teeth_impact.png) · [PDF](per_activity/07_brushing_teeth_impact.pdf)|[PNG](per_activity_ovr_macro_f1/07_brushing_teeth_impact.png) · [PDF](per_activity_ovr_macro_f1/07_brushing_teeth_impact.pdf)|
|eating soup|7/20|[PNG](per_activity/08_eating_soup_impact.png) · [PDF](per_activity/08_eating_soup_impact.pdf)|[PNG](per_activity_ovr_macro_f1/08_eating_soup_impact.png) · [PDF](per_activity_ovr_macro_f1/08_eating_soup_impact.pdf)|
|eating chips|6/20|[PNG](per_activity/09_eating_chips_impact.png) · [PDF](per_activity/09_eating_chips_impact.pdf)|[PNG](per_activity_ovr_macro_f1/09_eating_chips_impact.png) · [PDF](per_activity_ovr_macro_f1/09_eating_chips_impact.pdf)|
|eating pasta|9/20|[PNG](per_activity/10_eating_pasta_impact.png) · [PDF](per_activity/10_eating_pasta_impact.pdf)|[PNG](per_activity_ovr_macro_f1/10_eating_pasta_impact.png) · [PDF](per_activity_ovr_macro_f1/10_eating_pasta_impact.pdf)|
|drinking from cup|7/20|[PNG](per_activity/11_drinking_from_cup_impact.png) · [PDF](per_activity/11_drinking_from_cup_impact.pdf)|[PNG](per_activity_ovr_macro_f1/11_drinking_from_cup_impact.png) · [PDF](per_activity_ovr_macro_f1/11_drinking_from_cup_impact.pdf)|
|eating sandwich|6/20|[PNG](per_activity/12_eating_sandwich_impact.png) · [PDF](per_activity/12_eating_sandwich_impact.pdf)|[PNG](per_activity_ovr_macro_f1/12_eating_sandwich_impact.png) · [PDF](per_activity_ovr_macro_f1/12_eating_sandwich_impact.pdf)|
|kicking (soccer ball)|5/20|[PNG](per_activity/13_kicking__soccer_ball_impact.png) · [PDF](per_activity/13_kicking__soccer_ball_impact.pdf)|[PNG](per_activity_ovr_macro_f1/13_kicking__soccer_ball_impact.png) · [PDF](per_activity_ovr_macro_f1/13_kicking__soccer_ball_impact.pdf)|
|playing catch w/tennis ball|9/20|[PNG](per_activity/14_playing_catch_w_tennis_ball_impact.png) · [PDF](per_activity/14_playing_catch_w_tennis_ball_impact.pdf)|[PNG](per_activity_ovr_macro_f1/14_playing_catch_w_tennis_ball_impact.png) · [PDF](per_activity_ovr_macro_f1/14_playing_catch_w_tennis_ball_impact.pdf)|
|dribbling (basketball)|2/20|[PNG](per_activity/15_dribbling__basketball_impact.png) · [PDF](per_activity/15_dribbling__basketball_impact.pdf)|[PNG](per_activity_ovr_macro_f1/15_dribbling__basketball_impact.png) · [PDF](per_activity_ovr_macro_f1/15_dribbling__basketball_impact.pdf)|
|writing|6/20|[PNG](per_activity/16_writing_impact.png) · [PDF](per_activity/16_writing_impact.pdf)|[PNG](per_activity_ovr_macro_f1/16_writing_impact.png) · [PDF](per_activity_ovr_macro_f1/16_writing_impact.pdf)|
|clapping|7/20|[PNG](per_activity/17_clapping_impact.png) · [PDF](per_activity/17_clapping_impact.pdf)|[PNG](per_activity_ovr_macro_f1/17_clapping_impact.png) · [PDF](per_activity_ovr_macro_f1/17_clapping_impact.pdf)|
|folding clothes|5/20|[PNG](per_activity/18_folding_clothes_impact.png) · [PDF](per_activity/18_folding_clothes_impact.pdf)|[PNG](per_activity_ovr_macro_f1/18_folding_clothes_impact.png) · [PDF](per_activity_ovr_macro_f1/18_folding_clothes_impact.pdf)|

代码入口：`/home/wanzhen/attribute_prototype_analysis/analyze_initial9_loao_by_activity.py`。
复现命令：

```bash
bash /home/wanzhen/attribute_prototype_analysis/run_initial9_loao_by_activity.sh wisdm
```
