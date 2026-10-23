# 基于完整 A1 原型的语义属性细化

当前入口为 `src/zshar/semantics/planner1.py`，策略标识为
`full_a1_semantic_refinement_v3`。该版本只规划和验证 A2 语义，不在细化阶段绑定
特征组、selector 或具体特征列。

## 输入与证据

planner 接收最新 A1 活动向量、正式 A1 ontology、预计算 GAP CSV、真实特征名和
G1–G14 配置。GAP CSV 由 `src/zshar/semantics/gap_analysis.py` 生成，planner 直接
复用其中的：

- split 级和逐 unseen 的 D_kNN；
- split 级和逐 unseen 的 U_NN；
- 每个 unseen 的最近 seen 原型；
- top-k 和距离类型。

planner 不再重复计算这些指标，但仍校验 CSV 的 A1 哈希，防止旧诊断与新 A1 混用。

## 提供给 LLM 的信息

- zero-shot 数据集名称、split 编号、真实 seen/unseen 活动名；
- 完整 A1 ontology 和全部活动的 A1 原型矩阵；
- CSV 中已有的 D_kNN、U_NN、逐活动诊断和最近 seen 邻居；
- 当前数据集实际非空的 G1–G14 粗粒度物理能力；
- 数据集传感器位置及可用 acc/gyro 模态。

不提供样本、实测特征值、预测、准确率、具体特征列或 selector profile。

提示词要求重点比较 A1 中只相差一到两个坐标的活动对。高 D_kNN 表示 seen 语义
覆盖较弱；低 U_NN 表示 unseen 活动之间距离较近。两者都用于确定优先细化对象。
传感器模态、身体位置和运动轴可以成为细化角度，前提是有明确物理含义、当前数据集
能够观测，并能帮助区分易混活动。

新增的可预测性要求：候选属性的三级状态应对应传感器及现有特征能够观测、稳定且
可跨活动迁移的运动模式，不能把语义可区分性视为可预测性的保证。每个候选属性须在
对应操作的 `reason` 中说明支持特征组、预期传感器证据和预测歧义；多输出操作须逐个
说明。这是语义可行性评估，不生成特征绑定或活动级属性向量。不确定其可预测性或
新增区分价值时，优先保留已有 A1 属性及其定义。物理概念不重复、操作谱系与最多
15 个属性的既有约束继续适用。

## 输出与验证

LLM 只返回：

- A2 属性名、定义及 Low/Medium/High；
- keep/split/merge/drop/replace/add 操作谱系；
- split 诊断理由和语义冗余审查。

每个属性对象只允许 `name`、`definition` 和 `levels`。不允许返回
`primary_groups`、`auxiliary_groups`、`group_evidence` 或 `feature_selectors`。

本地验证检查属性数量、三级语义、名称格式、操作基数、A1/A2 完整谱系、seen/unseen
角色和 keep 语义冻结。单词属性名可以直接使用；多词属性名必须使用小写
`snake_case`。验证不禁止传感器、位置或轴相关概念，也不生成
`a2_attribute_plan.feature_mapping.json`。

## Dry run 示例

```bash
cd /home/wanzhen/myproject_new
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python \
  -m zshar.semantics.planner1 \
  --dataset-name pamap2 \
  --a1-json configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json \
  --a1-ontology-json configs/attributes/initial_attributes.json \
  --dknn-csv runtime/gap/pamap2/a1_gap_normalized_ordinal_seed42.csv \
  --split-index 13 \
  --output-json runtime/refinement_semantic_v3/pamap2/split_13/a2_attribute_plan.json \
  --min-attributes 11 --max-attributes 15 --dry-run
```

使用新版策略时应采用新的输出目录；planner 会拒绝把新策略产物写进旧绑定策略目录。

## A2 属性向量一次性完整生成

`src/zshar/semantics/completion.py` 的生成策略为 `full_a2_generation_v1`。
每个 split 在一次 LLM 请求中生成全部活动、全部最终 A2 属性的完整矩阵：

- `keep` 只保留概念定义和等级含义；每个活动的取值也重新生成，不复制 A1。
- `add/replace/split/merge` 统一按最终 A2 定义重新赋值；`drop` 不出现在输出中。
- 全部 `keep` 的方案和历史 `fallback_a1` 方案也必须请求完整矩阵，不直接复用 A1。
- A1 向量和 ontology 仅作为历史语义上下文，不是固定值或数值软锚点。
- 只返回变化列、缺少任一活动/属性或含非法等级时校验失败，程序不会从 A1 补齐。

报告记录所有 `llm_generated_attributes`，`copied_keep_attributes` 固定为空，
并保存 plan、A1、原始响应及输出向量哈希。原始响应旁的 `.generation.json`
记录生成策略和上下文哈希，供中断恢复校验使用。
PAMAP2、MHEALTH、WISDM 向量批处理及 MHEALTH 全流程不会将旧的复制/增量结果
作为新策略的完成检查点；恢复旧原始响应时也不能重新拼接 A1。

## 原型距离和重算

默认指标名为 `normalized_ordinal`，采用等权归一化序数原型距离：
`d_A(i,j) = (1/K_A) * sum_k(abs(a_i,k-a_j,k)/2)`，其中 L/M/H 为 0/1/2。
内部原有 0/0.5/1 编码与此公式严格等价；旧等权 `mad` 的指标不会因更名而改变。
新距离禁止引入非等权分类器可靠性权重。D_kNN、U_NN、U_Conf、JointGap 和最近
seen 邻居统一使用该距离，并以每个 split 的实际 A1/A2 属性数归一化。

A1 诊断默认入口和命令见 [A1 GAP](A1_GAP.md)。实际重算文件为
`runtime/gap/<dataset>/a1_gap_normalized_ordinal_seed42.csv`，批量 launcher 默认读取它。
保存向量的 A1/A2 对比可重新运行：

```bash
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python \
  -m zshar.semantics.gap_comparison \
  --dataset mhealth \
  --a2-vectors-root runtime/refinement_semantic_v3/mhealth_gpt_5_6_sol_11_15 \
  --output-dir runtime/gap/normalized_ordinal_seed42/mhealth
```

该命令导出 `a1_gap.csv`、`a2_gap.csv`、`comparison.csv`、`summary.json` 和 A1
元数据；只读取保存的原型，不调用 LLM 或重新训练分类器。当前 PAMAP2、MHEALTH、
WISDM 各 20 个 split 的重算基于既有向量，不能视为新生成策略的实验结果。
