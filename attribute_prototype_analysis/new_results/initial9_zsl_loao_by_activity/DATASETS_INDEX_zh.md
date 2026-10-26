# 初始9维属性：活动级LOAO结果索引

所有主图采用与PAMAP2相同的Top-1（类别召回率）和活动类别F1。差值=完整9维−删除后8维，单位pp。每个活动只在它作为Unseen出现的split上做配对统计，不补齐到20次。

只读取 initial9_zsl_loao 已保存的预测，不加载模型、不重新预测或训练。MHEALTH本次仍对应原始全High的motion_regularity；configs/attributes_new中的新语义标注不混入这批历史模型的消融结果。

|数据集|活动数|每活动Unseen次数|Top-1＋类别F1总览|逐活动图表索引|Excel|
|---|---:|---:|---|---|---|
|PAMAP2|12|4–15/20|[PNG](pamap2/activity_attribute_impact_top1_f1.png) · [PDF](pamap2/activity_attribute_impact_top1_f1.pdf)|[索引](pamap2/RESULT_INDEX_zh.md)|[Excel](pamap2/activity_loao_results.xlsx)|
|MHEALTH|12|4–12/20|[PNG](mhealth/activity_attribute_impact_top1_f1.png) · [PDF](mhealth/activity_attribute_impact_top1_f1.pdf)|[索引](mhealth/RESULT_INDEX_zh.md)|[Excel](mhealth/activity_loao_results.xlsx)|
|USC-HAD|12|4–12/20|[PNG](uschad/activity_attribute_impact_top1_f1.png) · [PDF](uschad/activity_attribute_impact_top1_f1.pdf)|[索引](uschad/RESULT_INDEX_zh.md)|[Excel](uschad/activity_loao_results.xlsx)|
|WISDM|18|2–10/20|[PNG](wisdm/activity_attribute_impact_top1_f1.png) · [PDF](wisdm/activity_attribute_impact_top1_f1.pdf)|[索引](wisdm/RESULT_INDEX_zh.md)|[Excel](wisdm/activity_loao_results.xlsx)|
|DSADS|19|3–11/20|[PNG](dsads/activity_attribute_impact_top1_f1.png) · [PDF](dsads/activity_attribute_impact_top1_f1.pdf)|[索引](dsads/RESULT_INDEX_zh.md)|[Excel](dsads/activity_loao_results.xlsx)|

每个数据集下 per_activity/ 包含各活动独立Top-1／类别F1双子图（PNG/PDF/SVG）；误差线为该活动对应split配对差值的样本标准差。
一对其余二分类Macro-F1是另一个补充口径，主图与本文表格均以class_f1为准。
datasets_summary.csv 汇总活动数、实际Unseen次数和总体指标重放数；各目录 independent_verification.json 记录逐类别指标与配对统计复核。

复现命令：

```bash
for ds in mhealth uschad wisdm dsads; do
  bash /home/wanzhen/attribute_prototype_analysis/run_initial9_loao_by_activity.sh "$ds"
done
```
