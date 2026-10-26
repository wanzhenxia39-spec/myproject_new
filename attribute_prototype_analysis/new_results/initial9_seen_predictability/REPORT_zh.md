# 9维属性的 Seen 验证集可预测性

数据集：PAMAP2、MHEALTH、USC-HAD、WISDM、DSADS。使用 initial v1.5 属性原型，seed=42，
每个数据集沿用 20 个 ZSL 类别划分及独立 AutoGluon 属性分类器。

完成 split 数：PAMAP2 20/20，MHEALTH 20/20，USC-HAD 20/20，WISDM 20/20，DSADS 20/20。

## Macro-F1：越高越好

| 属性 | PAMAP2 | MHEALTH | USC-HAD | WISDM | DSADS |
|---|---:|---:|---:|---:|---:|
| 运动强度 (`motion_intensity`) | 0.9689 | 1.0000 | 0.9974 | 0.9795 | 0.9948 |
| 动作频率 (`cadence`) | 0.9702 | 0.9998 | 0.9970 | 0.9795 | 0.9979 |
| 平滑性 (`smoothness`) | 0.9629 | 0.9934 | 0.9737 | 0.9780 | 0.9938 |
| 冲击性 (`impulsiveness`) | 0.9713 | 0.9911 | 0.9753 | 0.9875 | 0.9973 |
| 姿态倾斜度 (`posture_inclination`) | 0.9734 | 0.9975 | 0.9914 | 0.9884 | 0.9981 |
| 姿态变化幅度 (`posture_variability`) | 0.9710 | 0.9962 | 0.9948 | 0.9846 | 0.9936 |
| 旋转运动 (`rotational_motion`) | 0.9693 | 0.9996 | 0.9949 | 0.9682 | 0.9939 |
| 平移运动 (`translational_motion`) | 0.9801 | 0.9890 | 0.9885 | 0.9853 | 0.9974 |
| 运动规律性 (`motion_regularity`) | 0.9697 | 1.0000* | 0.9925 | 0.9691 | 0.9852 |

## EOD：越低越好

| 属性 | PAMAP2 | MHEALTH | USC-HAD | WISDM | DSADS |
|---|---:|---:|---:|---:|---:|
| 运动强度 (`motion_intensity`) | 0.0200 | 0.0076 | 0.0146 | 0.0157 | 0.0097 |
| 动作频率 (`cadence`) | 0.0209 | 0.0400 | 0.0214 | 0.0197 | 0.0515 |
| 平滑性 (`smoothness`) | 0.0203 | 0.0912 | 0.0317 | 0.0171 | 0.0140 |
| 冲击性 (`impulsiveness`) | 0.0247 | 0.1397 | 0.0342 | 0.0119 | 0.0262 |
| 姿态倾斜度 (`posture_inclination`) | 0.0184 | 0.0853 | 0.0105 | 0.0083 | 0.0304 |
| 姿态变化幅度 (`posture_variability`) | 0.0179 | 0.0376 | 0.0065 | 0.0112 | 0.0120 |
| 旋转运动 (`rotational_motion`) | 0.0253 | 0.0681 | 0.0145 | 0.0212 | 0.0263 |
| 平移运动 (`translational_motion`) | 0.0652 | 0.1014 | 0.0211 | 0.0085 | 0.0185 |
| 运动规律性 (`motion_regularity`) | 0.0180 | 0.0000* | 0.0216 | 0.0255 | 0.0103 |

两表均为每个属性在 20 个 split 上的等权平均，数值范围 [0,1]。标准差分别见
`macro_f1_std_9x5.csv`、`eod_std_9x5.csv`；标准差为类别划分间的描述性波动。

## 统计口径和验证集用途

- 每个 split 从 Seen 类窗口中按活动分层留出 30%，其余 70% 为外层训练池。
- 现有 AutoGluon 后端没有把这 30% 作为 tuning_data 传入 fit；它在 70% 内部再留出一部分数据用于模型、集成和二分类阈值选择。
- 逐模型检查缓存行索引和标签，要求内部拟合集与内部选模集的并集恰好为外层 70%，且均不与外层 30% 重合。
- 外层验证准确率曾用于原 ZSL 的活动匹配属性权重，但未用于选择属性分类器；本次直接计算属性预测指标，不执行活动匹配、重新训练、选模或阈值调优。
- 主 Macro-F1 与项目既有实现一致，对真实值与预测值的等级并集求平均，zero_division=0；固定对 Low/Medium/High 三等级求平均的备选矩阵见 `macro_f1_all_levels_mean_9x5.csv`。
- EOD 使用 Low=0、Medium=1、High=2，先对三等级的预测概率加权绝对距离，再对样本求平均，分母固定为 2。缺失训练等级的概率补 0，不压缩等级间距。
- 真实标签来自活动对应的设计属性原型，而非每个 IMU 窗口的独立物理测量标注。

## 单一等级与结论边界

带 * 的单元格包含至少一个仅有单一训练等级的 split，其高 F1 / 低 EOD 不能单独证明该属性能区分不同等级。具体数量见 `constant_split_count_9x5.csv`。

- MHEALTH / 运动规律性：20 个 split 为单一训练等级。

训练集多数类预测的 Macro-F1、训练等级分布的 EOD 均单独导出，便于与无信号基线比较。
结果支持对已有 Seen 活动语义属性标签的窗口级预测能力；不直接证明 Unseen 活动、独立受试者或独立采集记录上的泛化。现有窗口划分可能共享受试者和重叠原始记录。

## 文件

- `attribute_predictability_9x5.xlsx`：两项主指标、标准差、备选口径、基线及明细。
- `macro_f1_heatmap_9x5.{png,pdf,svg}` 和 `eod_heatmap_9x5.{png,pdf,svg}`：两张热力图。
- `per_split_attribute_metrics.csv`：100 个 split × 9 属性的逐 split 指标。
- `<dataset>/split_<index>/validation_predictions.csv.gz`：验证窗口、真值、预测等级、三等级概率和逐样本 EOD。
- 各 split 的 `partition.csv`、`audit.json`、`metrics.json`：划分、隔离检查和指标来源。
- `protocol.json`、`code_snapshot/` 和 `README.md`：协议、代码快照和重跑说明。

