# DSADS 初始9维属性：按活动统计的逐维移除实验

本次只读取 initial9_zsl_loao 已保存的逐窗口真实活动标签及10种配置的预测，不加载模型、不重新预测、不重新训练或选模。使用初始 v1.5 属性及原有20个ZSL split，候选集合为每个 split 的 Unseen 类。
主差值沿用原实验：完整9维 − 删除一维后的8维，单位为百分点（pp）。正值表示该维在当前条件下有正贡献；负值表示删除后更好。反方向“删除后 − 完整”同时保存在表格。

## 活动条件统计口径

对于活动 c，仅纳入它作为 Unseen 出现的 split 集合 S_c。先在每个 split 上分别计算完整/删除配置的指标并相减，再在同一 S_c 上等权求均值。不是将未出现该活动的 split 填0，也不是将所有活动均值无条件平均为原来的总体分数。
Top-1(c)=TP/(TP+FN)，即真实属于 c 的测试窗口的识别准确率（类别召回率）；所有错判均进入 FN。
单个活动的 F1(c)=2TP/(2TP+FP+FN)。FP 必须来自同一 split 中其他活动被预测成 c 的样本，因此先使用完整候选集合计算混淆计数，再提取该活动的 F1。不能仅筛选 y_true=c 的行后调用 macro F1。
原来的 Macro-F1 是该 split 各 Unseen 活动 F1 的均值；主图第二面板显示它对应的活动类别 F1，而不将其误称为单个活动的跨类别 Macro-F1。为满足另一种“每活动宏F1”定义，另提供严格的一对其余（二分类）Macro-F1=(F1(c)+F1(non-c))/2 的双子图和表格；它包含其余类表现，不能与原多类 Macro-F1 混用。
本次 DSADS 每个 split 的每个 Unseen 类固定30个测试窗口。误差线为 S_c 上配对差值的样本标准差（ddof=1），不是置信区间；总览条形矩阵不画误差线，逐活动图和Excel提供标准差。n=1不计算标准差。split共享数据，因此不能将这些均值或标准差视为独立重复的显著性检验。
完整/删除后的活动均值采用相同split集合，两者均值相减等于配对差值的均值。各活动的n不同，因此将各活动条件均值再简单平均，不等于原20个split的总体均值；若要还原总体指标，应按各活动出现的split次数加权（本数据每split类别数与每类测试窗口数固定）。

## 每个活动实际作为 Unseen 的次数

|活动|n/20|窗口出现次数|split索引|
|---|---:|---:|---|
|sitting|11|330|820,977,1042,2849,3071,3359,3649,4573,7165,7315,8025|
|standing|5|150|820,977,1042,9013,13826|
|lying on back|6|180|2849,3071,3359,3649,9013,17871|
|lying on right side|6|180|820,977,4573,9013,19350,20953|
|ascending stairs|5|150|1042,2849,3071,19350,22175|
|descending stairs|10|300|1042,2849,3359,4573,7165,19350,22175,24133,24271,24300|
|standing in an elevator still|10|300|3649,4573,7315,13826,17871,20953,22175,24133,24271,24300|
|moving around in an elevator|10|300|820,2849,4573,7315,8025,17871,20953,22175,24133,24271|
|walking in a parking lot|7|210|1042,3071,3359,7315,17871,24133,24300|
|walking on a treadmill with a speed of 4 km/h (in flat)|4|120|2849,13826,24133,24300|
|walking on a treadmill with a speed of 4 km/h (in 15 deg inclined positions)|6|180|820,3649,7165,8025,19350,24300|
|running on a treadmill with a speed of 8 km/h|3|90|977,7165,20953|
|exercising on a stepper|3|90|1042,20953,24271|
|exercising on a cross trainer|7|210|7165,7315,8025,9013,13826,17871,24133|
|cycling on an exercise bike in horizontal positions|3|90|3071,3359,22175|
|cycling on an exercise bike in vertical positions|3|90|977,19350,24271|
|rowing|7|210|820,3071,4573,8025,13826,20953,24300|
|jumping|8|240|3649,7165,7315,9013,13826,17871,19350,22175|
|playing basketball|6|180|977,3359,3649,8025,9013,24271|

## Top-1：完整9维 − 删除后的平均差值（pp）

|属性|sitting|standing|lying on back|lying on right side|ascending stairs|descending stairs|standing in an elevator still|moving around in an elevator|walking in a parking lot|walking on a treadmill with a speed of 4 km/h (in flat)|walking on a treadmill with a speed of 4 km/h (in 15 deg inclined positions)|running on a treadmill with a speed of 8 km/h|exercising on a stepper|exercising on a cross trainer|cycling on an exercise bike in horizontal positions|cycling on an exercise bike in vertical positions|rowing|jumping|playing basketball|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|motion_intensity|+0.606|+1.333|+0.000|+0.556|-8.000|+2.333|+16.667|-1.000|+16.667|+0.000|-1.111|+1.111|-4.444|-5.714|+0.000|+0.000|+5.714|+5.833|+2.778|
|cadence|+0.000|+2.667|+0.000|+0.000|+2.667|+8.333|+13.000|-1.667|+9.048|+0.000|+10.556|-11.111|+5.556|+0.000|+0.000|+32.222|+2.381|+2.917|+5.000|
|smoothness|+25.152|+1.333|+0.000|-46.667|+15.333|-25.000|+0.333|-0.333|+9.524|-5.000|+3.889|-13.333|+11.111|+1.905|+0.000|+0.000|-2.857|+3.750|+0.556|
|impulsiveness|-0.606|-1.333|+0.000|+0.000|+27.333|-38.000|-8.333|+1.667|+28.095|+1.667|+10.000|-7.778|-1.111|+2.381|+0.000|+0.000|-2.381|+7.917|-3.889|
|posture_inclination|-21.515|+58.000|+73.333|+1.111|-15.333|+3.667|+4.000|+3.000|+7.143|+0.000|-9.444|+3.333|+6.667|+2.857|+0.000|+0.000|+0.476|+4.167|+2.222|
|posture_variability|+0.000|+2.000|+0.000|+0.000|+2.667|+5.000|+0.000|-1.000|+0.000|+0.000|+1.111|-2.222|+24.444|+0.952|+0.000|+0.000|-37.619|+3.333|-3.889|
|rotational_motion|+0.000|+0.000|+0.000|+0.000|+10.000|+3.333|+20.000|-0.333|+8.095|+0.833|+8.333|+0.000|-1.111|+8.095|+0.000|+0.000|+8.571|+0.000|-1.111|
|translational_motion|+0.606|+0.667|+0.000|+0.000|-1.333|+12.667|-12.333|-3.333|-0.952|+1.667|+3.889|-1.111|+5.556|+3.810|+0.000|+0.000|+2.857|-23.333|-7.222|
|motion_regularity|+0.606|-2.667|+0.000|+0.000|+4.000|+14.333|-7.000|-5.000|-8.571|+0.000|+13.889|+0.000|+3.333|+0.000|+0.000|+0.000|+0.000|+7.917|+2.778|

## 活动类别 F1：完整9维 − 删除后的平均差值（pp）

|属性|sitting|standing|lying on back|lying on right side|ascending stairs|descending stairs|standing in an elevator still|moving around in an elevator|walking in a parking lot|walking on a treadmill with a speed of 4 km/h (in flat)|walking on a treadmill with a speed of 4 km/h (in 15 deg inclined positions)|running on a treadmill with a speed of 8 km/h|exercising on a stepper|exercising on a cross trainer|cycling on an exercise bike in horizontal positions|cycling on an exercise bike in vertical positions|rowing|jumping|playing basketball|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|motion_intensity|-0.826|+2.536|-0.107|-1.989|-5.309|+3.855|+13.010|-1.152|+12.544|-3.100|-1.714|+0.903|+15.089|+4.936|+0.294|-3.294|+4.073|+7.139|+3.875|
|cadence|+0.978|-1.826|+0.000|-0.024|+0.914|+6.617|+7.449|+4.185|+4.427|+10.130|+7.149|-6.481|+12.202|-2.666|+5.042|+39.407|+1.175|+1.450|+7.742|
|smoothness|+14.907|+0.697|+0.000|-25.805|+10.960|-15.746|+0.331|-1.354|+3.688|-1.776|+2.538|-12.867|+0.430|-1.925|-0.546|-0.641|-3.286|+4.904|+0.518|
|impulsiveness|+0.184|-3.209|-0.560|-6.244|+17.644|-25.617|-7.753|+1.148|+18.842|-1.511|+2.759|-6.878|+2.583|+8.734|+0.184|+0.674|-2.098|+7.954|-4.486|
|posture_inclination|+0.571|+52.709|+61.964|+3.184|-4.489|+5.501|+3.442|+3.235|+3.010|-1.119|-1.586|+4.155|+0.523|-1.574|+8.511|-2.397|+2.577|+3.057|+2.963|
|posture_variability|-0.181|+0.623|+0.000|+0.261|+5.476|+3.338|-0.084|-2.336|+0.653|-1.292|+0.104|-3.915|+17.565|-2.064|-3.922|+2.273|-33.237|-0.589|-4.269|
|rotational_motion|+0.402|+1.332|+0.000|-5.123|+7.691|+2.235|+17.920|-0.041|+4.902|-0.982|+4.478|+0.000|+20.475|+6.840|+0.529|+0.904|+9.631|+0.466|-1.955|
|translational_motion|+0.493|-0.257|-0.680|-5.099|+0.246|+7.596|-11.783|-0.630|-0.866|+2.925|+1.480|-2.116|-1.152|+5.299|-2.317|+0.053|+5.586|-18.664|-6.541|
|motion_regularity|+0.311|-2.526|-1.319|-6.140|+1.582|+10.300|-10.080|+3.621|+0.261|+8.267|+5.360|-0.345|-4.347|-2.686|-1.357|+0.000|-0.794|+5.286|+3.820|

## 文件入口

- activity_attribute_impact_top1_f1.png/pdf/svg：纵轴9维属性、横轴活动类别的条形矩阵总览，上下两个面板分别为Top-1与活动类别F1。每列灰线为零点，条形向左/右表示负/正差值；各面板内部所有条形采用相同长度尺度。
- per_activity/：每个活动单独一张Top-1 / 类别F1双子图，共19张（PNG/PDF/SVG），带配对标准差。
- activity_attribute_impact_top1_ovr_macro_f1.* 和 per_activity_ovr_macro_f1/：严格二分类Macro-F1替代口径，含义与主图类别F1不同。
- activity_loao_results.xlsx：绝对值、双方向差值、标准差、活动覆盖次数、逐split分类混淆计数及总体重放核对。
- activity_attribute_summary.csv：每活动×每属性的完整值、删除后值、差值、标准差与正负split数。
- per_split_activity_metrics.csv / paired_activity_deltas.csv：仅该活动作为Unseen时的逐split结果与配对差值。
- source_audit.json / verification.json：输入文件哈希和核验结果。

该实验评价冻结系统中的条件边际影响，不能将各单维删除收益相加，也不能依据测试消融选择属性子集后将同一测试集当作独立验证。
