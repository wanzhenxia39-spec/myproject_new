# 最新版 A2 v4 细化属性逐维移除实验

差值定义：完整细化属性集 − 删除该属性后的结果，单位为百分点（pp）。正值表示删除导致下降，负值表示删除后改善。
完整基线是每个 split 各自的完整 A2 属性集（9–12 维），不是先裁成公共维度。每次只从该完整基线移除一维，其他维度、验证准确率权重、候选类别和测试窗口固定；剩余权重重新归一化。
本次沿用每个 split 已保存的独立属性模型，不重训、不重新选模；完整配置和每个删除配置都独立重新加载全部保留分类器并调用 predict_proba，概率和原型同时去除该维后重新执行序数概率匹配。
使用 configs/attrIbutes_v4 下最新 full_a2_generation_v1 的原型；全部100份原型 SHA256 与对应已完成模型运行一致。DSADS 使用 refined_rerun_20261021；其余均使用 seed42 的 A2 v4 independent t300 运行。
所有 split 使用固定 ZSL Unseen 测试窗口与仅 Unseen 的候选类别。Macro-F1 为活动分类指标。模型和权重在本次测试前已冻结。

## 不同 split 属性集不同：三个统计层次

1. 主图：所有100个 split 共同保留且定义与等级描述完全一致的属性：motion_intensity, cadence, smoothness, impulsiveness。每个数据集每个条形均基于20个完整配对差值。
2. 数据集公共属性图：各数据集自身20个 split 的交集。可评价在本数据集稳定保留、但不属于全局交集的属性。
3. 精确定义补充图（exact_semantics）：所有实际出现的属性按“名称＋定义＋三级等级描述”精确分组。定义变化时不混合平均，标签 s 标明对应 split；标注 n，未出现不填0，n=1 不计算标准差。新增或拆分属性的不同名称不强行合并。
同名合并图（all_attributes）仅作为探索性索引，可能混合不同定义，不能作为属性必要性的主证据。定义变化数见 semantic_variants，精确定义及 split 清单见 exact_semantics_catalog.csv。条件子集的均值只能解释该子集，不能与20个 split 的均值直接作为公平排名；不同完整属性上下文也可能影响删除贡献。
误差线为各 split 配对差值的样本标准差（ddof=1），不是置信区间。20个 split 共享数据，不能据此声称统计显著或独立重复。测试消融属于事后分析，不应再用这些测试结果选择部署属性子集。

## 完整细化基线（20 splits）

|数据集|维数|Top-1 (%)|Macro-F1 (%)|
|---|---|---:|---:|
|PAMAP2|10–10|55.70 ± 14.20|47.68 ± 17.03|
|MHEALTH|9–11|63.88 ± 17.31|55.13 ± 21.28|
|USC-HAD|9–10|54.33 ± 11.91|45.98 ± 14.25|
|WISDM|10–11|38.22 ± 13.33|32.23 ± 13.32|
|DSADS|10–12|51.89 ± 15.38|44.16 ± 17.08|

## Top-1：全局公共属性差值 (pp)

|属性|PAMAP2|MHEALTH|USC-HAD|WISDM|DSADS|
|---|---:|---:|---:|---:|---:|
|motion_intensity|+0.525|+1.458|+1.000|+1.806|+0.333|
|cadence|+0.550|-2.292|-0.542|+0.944|+1.194|
|smoothness|-0.575|-0.458|+0.583|-0.056|-0.194|
|impulsiveness|+3.150|-1.458|-0.833|+1.306|-0.444|

全局公共属性共 20 个属性×数据集组合，其中负均值 9 个；各数据集自身20个 split 公共属性共 39 组，其中负均值 17 个。这两层均有完整20个配对结果，是本次主要分析依据。

探索性同名属性×数据集条件均值：129 项中，负值 52 项。以下仅作检索（含实际出现 split 数），不同定义混合的组不用于必要性结论：

|数据集|属性|n/20|差值 pp|
|---|---|---:|---:|
|USC-HAD|cadence|20|-0.542|
|USC-HAD|impulsiveness|20|-0.833|
|USC-HAD|posture_inclination|20|-0.708|
|USC-HAD|motion_regularity|20|-0.208|
|USC-HAD|vertical_motion_prominence|1|-1.667|
|USC-HAD|vertical_motion_dominance|3|-0.833|
|USC-HAD|vertical_dynamic_emphasis|1|-0.833|
|MHEALTH|cadence|20|-2.292|
|MHEALTH|smoothness|20|-0.458|
|MHEALTH|impulsiveness|20|-1.458|
|MHEALTH|posture_variability|18|-2.546|
|MHEALTH|rotational_motion|16|-0.104|
|MHEALTH|translational_motion|20|-1.458|
|MHEALTH|trunk_reorientation|1|-5.000|
|MHEALTH|lower_leg_inclination|1|-0.833|
|MHEALTH|lower_leg_orientation_variability|3|-0.833|
|MHEALTH|arm_motion_dominance|1|-1.667|
|MHEALTH|upper_limb_rotational_motion|1|-24.167|
|PAMAP2|smoothness|20|-0.575|
|PAMAP2|posture_variability|20|-0.150|
|PAMAP2|translational_motion|20|-0.925|
|PAMAP2|hand_motion_intensity|1|-9.500|
|PAMAP2|lower_leg_inclination|2|-1.250|
|PAMAP2|trunk_inclination|2|-9.750|
|PAMAP2|hand_cyclic_motion|1|-0.500|
|PAMAP2|cyclic_hand_motion_magnitude|1|-14.000|
|PAMAP2|lower_limb_motion_intensity|1|-5.500|
|PAMAP2|ankle_rotational_motion|2|-5.250|
|PAMAP2|lower_leg_rotational_motion|1|-3.000|
|PAMAP2|hand_relative_motion|1|-2.000|
|PAMAP2|limb_motion_balance|1|-1.000|
|PAMAP2|hand_motion_extent|1|-7.500|
|WISDM|smoothness|20|-0.056|
|WISDM|posture_inclination|20|-0.611|
|WISDM|rotational_motion|20|-0.444|
|WISDM|wrist_inclination_excursion|2|-3.056|
|DSADS|smoothness|20|-0.194|
|DSADS|impulsiveness|20|-0.444|
|DSADS|translational_motion|18|-0.309|
|DSADS|motion_regularity|20|-0.056|
|DSADS|upper_limb_motion|2|-5.278|
|DSADS|intensity_variability|2|-0.278|
|DSADS|torso_inclination|2|-1.667|
|DSADS|arm_motion_prominence|1|-3.333|
|DSADS|upper_limb_motion_amplitude|1|-1.667|
|DSADS|vertical_directionality|1|-6.667|
|DSADS|whole_body_coupling|1|-0.556|
|DSADS|impact_pattern|1|-3.889|
|DSADS|upper_limb_involvement|1|-11.111|
|DSADS|lateral_torso_inclination|3|-1.111|
|DSADS|leg_motion_alternation|1|-1.667|
|DSADS|upper_limb_motion_intensity|1|-1.667|

## Macro-F1：全局公共属性差值 (pp)

|属性|PAMAP2|MHEALTH|USC-HAD|WISDM|DSADS|
|---|---:|---:|---:|---:|---:|
|motion_intensity|+0.868|+2.268|+1.209|+2.148|+0.328|
|cadence|+0.949|-2.542|-0.681|+0.667|+1.126|
|smoothness|-1.057|-0.619|+0.459|+0.597|-0.678|
|impulsiveness|+3.472|-1.921|-1.166|+1.647|-0.523|

全局公共属性共 20 个属性×数据集组合，其中负均值 8 个；各数据集自身20个 split 公共属性共 39 组，其中负均值 15 个。这两层均有完整20个配对结果，是本次主要分析依据。

探索性同名属性×数据集条件均值：129 项中，负值 54 项。以下仅作检索（含实际出现 split 数），不同定义混合的组不用于必要性结论：

|数据集|属性|n/20|差值 pp|
|---|---|---:|---:|
|USC-HAD|cadence|20|-0.681|
|USC-HAD|impulsiveness|20|-1.166|
|USC-HAD|posture_inclination|20|-0.567|
|USC-HAD|motion_regularity|20|-0.374|
|USC-HAD|vertical_motion_prominence|1|-1.948|
|USC-HAD|vertical_motion_dominance|3|-1.495|
|USC-HAD|vertical_dynamic_emphasis|1|-0.893|
|MHEALTH|cadence|20|-2.542|
|MHEALTH|smoothness|20|-0.619|
|MHEALTH|impulsiveness|20|-1.921|
|MHEALTH|posture_variability|18|-3.118|
|MHEALTH|rotational_motion|16|-0.010|
|MHEALTH|translational_motion|20|-1.935|
|MHEALTH|trunk_reorientation|1|-9.524|
|MHEALTH|lower_leg_inclination|1|-1.146|
|MHEALTH|lower_leg_orientation_variability|3|-1.104|
|MHEALTH|arm_motion_dominance|1|-1.784|
|MHEALTH|upper_limb_rotational_motion|1|-15.917|
|PAMAP2|smoothness|20|-1.057|
|PAMAP2|posture_variability|20|-0.085|
|PAMAP2|translational_motion|20|-1.288|
|PAMAP2|hand_motion_intensity|1|-12.777|
|PAMAP2|lower_leg_inclination|2|-1.525|
|PAMAP2|trunk_inclination|2|-13.979|
|PAMAP2|hand_cyclic_motion|1|-0.489|
|PAMAP2|cyclic_hand_motion_magnitude|1|-18.045|
|PAMAP2|lower_limb_motion_intensity|1|-7.416|
|PAMAP2|ankle_rotational_motion|2|-8.041|
|PAMAP2|lower_leg_rotational_motion|1|-2.974|
|PAMAP2|hand_relative_motion|1|-2.218|
|PAMAP2|limb_motion_balance|1|-0.849|
|PAMAP2|hand_motion_extent|1|-8.389|
|WISDM|posture_inclination|20|-0.715|
|WISDM|wrist_orientation_excursion|7|-0.028|
|WISDM|wrist_inclination_excursion|2|-2.468|
|DSADS|smoothness|20|-0.678|
|DSADS|impulsiveness|20|-0.523|
|DSADS|translational_motion|18|-0.415|
|DSADS|motion_regularity|20|-0.096|
|DSADS|upper_limb_motion|2|-4.043|
|DSADS|torso_lateral_inclination|4|-0.228|
|DSADS|intensity_variability|2|-0.126|
|DSADS|torso_inclination|2|-1.417|
|DSADS|arm_motion_prominence|1|-4.611|
|DSADS|upper_limb_motion_amplitude|1|-3.220|
|DSADS|motion_intensity_variability|1|-0.687|
|DSADS|lateral_posture|2|-0.058|
|DSADS|vertical_directionality|1|-10.266|
|DSADS|whole_body_coupling|1|-0.457|
|DSADS|impact_pattern|1|-2.904|
|DSADS|upper_limb_involvement|1|-13.262|
|DSADS|lateral_torso_inclination|3|-2.051|
|DSADS|leg_motion_alternation|1|-2.534|
|DSADS|upper_limb_motion_intensity|1|-2.221|

## 解释边界

负均值表示在当前冻结分类器、原型、匹配权重及测试划分下，该属性的加入使对应平均指标低于单维删除后；零均值也可能由正负效果抵消，需查看逐 split 明细。不能由一个负均值断言属性的语义设计无用，也不能将多个单维删除收益相加。
仅统计全局交集会漏掉细化引入的属性，因此主图与完整补充图一起解读。新增属性仅在部分 split 出现，样本较少，不能据其条件均值推出对全部 split 的普遍必要性。

## 完整性核对

```json
{
  "complete": true,
  "splits": 100,
  "configurations": 1105,
  "expected_configurations": 1105,
  "fresh_model_load_calls": 10145,
  "expected_fresh_calls": 10145,
  "fresh_predict_proba_calls": 10145,
  "autogluon_calls": 9901,
  "constant_calls": 244,
  "historical_full_baselines_exactly_reproduced": 100,
  "common_attributes": [
    "motion_intensity",
    "cadence",
    "smoothness",
    "impulsiveness"
  ],
  "drop_configurations_identical_probabilities_to_fresh_full": 1005,
  "drop_configurations": 1005
}
```
