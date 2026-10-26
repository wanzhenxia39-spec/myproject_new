# MHEALTH motion_regularity 重标注及五数据集覆盖率更新

原始 initial v1.5 MHEALTH 的12个活动均为 High，因而该维不同等级活动对为0/66。
新版保持属性名称、定义、Low/Medium/High等级说明和其余八维不变，仅将四类分次往返动作标为Medium：waist bends forward、frontal elevation of arms、knees bending (crouching)、jump front & back。其余八类为High；不为了增加覆盖率强行加入Low。
[UCI活动及采集协议](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) 支持四类动作各采集20次这一事实，但不直接决定其时间规律性。Medium依据分次动作允许中等节拍/相位变化的操作性语义规则，是人工审阅的先验假设；没有实测节拍、观察停顿或重新计算属性可预测性。静止状态的时间模式可稳定预测，因此不机械地标为Low。
原始配置、运行配置、模型、历史分类/可预测性结果保持不变。本次没有LLM请求、IMU推理、训练或测试分数调优。

## 属性区分覆盖率

R_k = 不同属性等级的无序活动对数 / C(C−1)/2。只判断等级是否不同，不按等级距离加权；与分类准确率、可预测性和属性互补性不同。本次覆盖全部活动原型，不基于20个split作模型评价。
MHEALTH 新版等级计数：High=8、Medium=4、Low=0。不同等级活动对=8×4=32；总活动对=12×11/2=66。因此 R=32/66=0.484848（48.4848%），原值为0。
五数据集9×5矩阵中仅 MHEALTH × motion_regularity 一个单元格变化；该属性跨5数据集的等权平均增加9.6970个百分点。
覆盖率上升是更改语义原型的直接结果，不证明新标注更符合真实IMU，也不证明分类性能或属性可预测性提高。若后续采用新目标标签训练，该属性需要重新训练和验证；旧常量分类器的结果不能直接套用。

## 最新9×5矩阵（0–1）

|属性|PAMAP2|MHEALTH|USC-HAD|WISDM|DSADS|
|---|---:|---:|---:|---:|---:|
|motion_intensity|0.712121|0.712121|0.712121|0.686275|0.695906|
|cadence|0.712121|0.621212|0.621212|0.627451|0.649123|
|smoothness|0.681818|0.666667|0.681818|0.627451|0.602339|
|impulsiveness|0.681818|0.621212|0.712121|0.568627|0.678363|
|posture_inclination|0.590909|0.621212|0.530303|0.503268|0.561404|
|posture_variability|0.409091|0.545455|0.621212|0.503268|0.584795|
|rotational_motion|0.681818|0.712121|0.666667|0.601307|0.695906|
|translational_motion|0.545455|0.621212|0.530303|0.385621|0.643275|
|motion_regularity|0.409091|0.484848|0.530303|0.660131|0.374269|

## MHEALTH逐活动标注

|活动|原值|新版|理由|
|---|---|---|---|
|standing still|High|High|静止姿态的时间模式稳定、可预测；规律性不等同于存在周期运动。|
|sitting and relaxing|High|High|保持坐姿的时间模式总体稳定；不因运动强度低而降低规律性。|
|lying down|High|High|保持卧姿的时间模式稳定、可预测。|
|walking|High|High|持续交替步态采用较稳定的重复时间模式。|
|climbing stairs|High|High|连续上楼的重复步态采用较稳定的时间模式，不预设停顿或转向。|
|waist bends forward|High|Medium|分次弯腰—恢复的往返动作有明确结构，但各次节拍及端点转换允许中等变化。此为语义标注假设。|
|frontal elevation of arms|High|Medium|分次抬臂—放下的往返动作有结构，但启动、回落和各次间隔允许中等变化。此为语义标注假设。|
|knees bending (crouching)|High|Medium|分次屈膝—起身动作有结构，但下降、转折、上升及各次间隔允许中等变化。此为语义标注假设。|
|cycling|High|High|持续踩踏采用重复且较稳定的周期时间模式。|
|jogging|High|High|持续慢跑采用较稳定的重复步态时间模式。|
|running|High|High|持续跑步采用重复步态；强冲击本身不等同于不规律。|
|jump front & back|High|Medium|前后交替跳有结构，但方向转换、落地与再次起跳节拍允许中等变化。此为语义标注假设。|

## 结果入口

- attribute_coverage_latest_9x5.png/pdf/svg：最新矩阵图。
- attribute_coverage_comparison_9x5.png/pdf/svg：原版/新版并排对照。
- attribute_coverage_change_9x5.png/pdf/svg：新版减原版，格内标注百分点。
- attribute_coverage_updated_9x5.xlsx：矩阵、差值、逐活动标注、精确配对计数及跨数据集均值。
- 新版五份向量及定义文件位于 `/home/wanzhen/myproject_new/configs/attributes_new/`。该路径指向数据盘上的项目目录；本组分析结果本身保存在系统盘。
