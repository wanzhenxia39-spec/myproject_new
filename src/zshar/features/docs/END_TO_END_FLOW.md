# ZSHAR 从 IMU 到活动标签的完整流程

下图按当前项目代码展示两条汇合的数据流：传感器窗口及其统计特征构成模型输入；
活动级语义属性向量构成属性监督和活动原型。ZSL 与 GZSL 可以按 split 使用 A2，
跨 subject 当前使用数据集配置中的公共属性向量。

```mermaid
flowchart TB
    subgraph PRE["① IMU 预处理与窗口特征"]
        RAW["原始可穿戴传感器流<br/>三轴加速度 / 三轴陀螺仪"]
        CLEAN["数据集预处理<br/>通道选择、缺失值处理、降采样"]
        WIN["按连续活动段切分固定长度窗口<br/>保存 window_index / activity_id / user_id"]
        FE["逐窗口特征抽取<br/>时域、频域、STFT、jerk、ACF、物理特征"]
        FMAT["特征矩阵 X<br/>及 feature_names"]
        RAW --> CLEAN --> WIN --> FE --> FMAT
    end

    subgraph SEM["② 离线语义资产：A1 到 split-specific A2"]
        ONT["A1 ontology<br/>9个属性的定义与 Low/Medium/High"]
        A1V["数据集活动级 A1 向量<br/>每个活动对应9维语义原型"]
        ASPLIT["活动 split manifest<br/>seen 类 S / unseen 类 U"]
        EV["完整 A1 原型 + 真实 seen/unseen<br/>计算语义 D_kNN 与最近邻 seen<br/>不筛选 issue cards"]
        GPHY["G1-G14 粗粒度物理能力词表<br/>只保留实际非空组"]
        CONCEPT["LLM 阶段1：概念细化<br/>keep / split / merge / drop / replace / add<br/>默认1-25个属性，允许原样保留 A1"]
        PRIOR["initial_attributes_group.json<br/>A1 属性到组的软先验"]
        GDETAIL["实际 feature_names + G1-G14<br/>位置、模态、轴的可用范围"]
        BIND["LLM 阶段2：属性与特征组绑定<br/>primary / auxiliary groups + selectors"]
        REVIEW{"本地逐属性审核<br/>语义来源完整且绑定非空？"}
        REPLAN["最多一次局部 replan + rebind"]
        FALLBACK["失败时回退 A1<br/>9属性及原始绑定"]
        A2PLAN["最终属性方案<br/>a2_attribute_plan.json"]
        MAP["精确绑定审计产物<br/>feature_mapping.json"]
        COMPLETE["LLM 属性向量补全<br/>只输入最终属性定义/等级和活动名称"]
        A2V["split-specific 活动级 A2 原型<br/>activity_attribute_vectors_a2.json"]

        ONT --> EV
        A1V --> EV
        ASPLIT --> EV
        EV -->|"完整矩阵、类名及语义距离送入提示词"| CONCEPT
        ONT --> CONCEPT
        GPHY --> CONCEPT
        EV --> BIND
        CONCEPT --> BIND
        PRIOR --> BIND
        GDETAIL --> BIND
        BIND --> REVIEW
        REVIEW -->|"通过"| A2PLAN
        REVIEW -->|"存在 rejected 属性"| REPLAN
        REPLAN -->|"复审通过"| A2PLAN
        REPLAN -->|"仍不通过"| FALLBACK
        FALLBACK --> A2PLAN
        A2PLAN --> MAP
        A2PLAN --> COMPLETE --> A2V
    end

    subgraph TABLE["③ 构造窗口级训练表"]
        LOAD["读取 X、feature_names、activity_id、user_id"]
        CHOOSE{"属性原型来源"}
        LABEL["按窗口真实 activity_name 查活动原型<br/>附加每个属性的 Low/Medium/High 监督标签"]
        DATA["DatasetTable<br/>元数据 + 全部特征列 + 属性标签"]
        FMAT --> LOAD
        LOAD --> LABEL
        CHOOSE --> LABEL
        LABEL --> DATA
    end

    A1V -->|"未指定 A2 root 或跨 subject"| CHOOSE
    A2V -->|"ZSL/GZSL 指定 attribute_vectors_root"| CHOOSE
    MAP -. "当前只保存和校验；runner 未用它裁剪训练列" .-> DATA

    SCENE{"④ 选择实验场景"}
    DATA --> SCENE

    subgraph ZSL["⑤A Zero-shot"]
        ZSPLIT["按活动划分<br/>训练：S 的窗口<br/>测试：U 的均衡抽样窗口"]
        ZTRAIN["seen 训练集内部按活动分层<br/>70% train / 30% validation"]
        ZMODEL["训练属性预测器<br/>输入为全部窗口特征 X<br/>输出 A1/A2 属性等级"]
        ZTEST["unseen 测试窗口 X"]
        ZATTR["预测属性向量<br/>硬标签或 Low/Medium/High 概率"]
        ZMATCH["只与 unseen 原型 U 比较<br/>cosine / euclidean / manhattan / ordinal probability"]
        ZOUT["Top-1 活动标签 + Top-3 排名<br/>accuracy / top3 recall / macro F1"]
        ZSPLIT --> ZTRAIN --> ZMODEL
        ZMODEL --> ZTEST --> ZATTR --> ZMATCH --> ZOUT
    end

    subgraph GZSL["⑤B Generalized zero-shot"]
        GSPLIT["按活动划分<br/>训练：S 的窗口<br/>测试：S∪U 的按活动抽样窗口<br/>测试 window_index 从训练中排除"]
        GTRAIN["seen 训练集内部按活动分层<br/>70% train / 30% validation"]
        GMODEL["训练属性预测器<br/>输入为全部窗口特征 X"]
        GCAL["可选 source-only 校准<br/>从 S 构造 pseudo-seen / pseudo-unseen folds<br/>选择 temperature τ 和 seen penalty γ"]
        GTEST["seen + unseen 测试窗口 X"]
        GATTR["预测属性向量或属性概率"]
        GMATCH["与 S∪U 全部活动原型比较"]
        GBIAS["GZSL 偏置修正<br/>温度缩放并对 seen 分数减 γ<br/>或仅执行 seen penalty"]
        GOUT["Top-1 活动标签 + Top-3 排名<br/>seen acc / unseen acc / H / overall acc<br/>macro F1 / 可选 AUSUC"]
        GSPLIT --> GTRAIN --> GMODEL
        GTRAIN --> GCAL
        GMODEL --> GTEST --> GATTR --> GMATCH --> GBIAS --> GOUT
        GCAL --> GBIAS
    end

    subgraph CS["⑤C 跨 subject"]
        SSPLIT["subject manifest 显式窗口索引<br/>训练 subject 与测试 subject 完全不重叠"]
        STRAIN["训练：train subjects 的全部有效活动窗口<br/>内部按活动分层 70% / 30%"]
        SMODEL["训练属性预测器<br/>公共 A1 属性监督"]
        STEST["测试：held-out subjects<br/>每个 subject×activity 等量抽样"]
        SATTR["预测属性向量或属性概率"]
        SMATCH["与全部活动的公共原型比较<br/>不做 GZSL seen penalty"]
        SOUT["Top-1 活动标签 + Top-3 排名<br/>accuracy / top3 recall / macro F1<br/>并输出 per-subject 指标"]
        SSPLIT --> STRAIN --> SMODEL --> STEST --> SATTR --> SMATCH --> SOUT
    end

    SCENE -->|"scenario=zsl"| ZSPLIT
    SCENE -->|"scenario=gzsl"| GSPLIT
    SCENE -->|"scenario=cross_subject"| SSPLIT

    ASPLIT --> ZSPLIT
    ASPLIT --> GSPLIT

    subgraph CHAIN["⑥ 属性预测器内部：以 autogluon_multilabel 硬标签链为例"]
        C1["第1个属性模型：X → Â1"]
        C2["第2个属性模型：X + Â1 → Â2"]
        CN["第k个属性模型：X + 前序预测属性 → Âk"]
        ORDER["split-specific forward / reverse / LLM order<br/>或后端默认顺序"]
        C1 --> C2 --> CN
        ORDER --> C1
    end

    ZMODEL -. "若选择硬标签链后端" .-> CHAIN
    GMODEL -. "若选择硬标签链后端" .-> CHAIN
    SMODEL -. "若选择硬标签链后端" .-> CHAIN
```

## 三个场景的核心区别

| 场景 | 属性模型训练窗口 | 测试窗口 | 活动候选集合 | 当前属性原型 |
|---|---|---|---|---|
| ZSL | 仅 seen 活动；不按 subject 隔离 | 仅 unseen 活动 | 仅 unseen 活动 | 可按 activity split 使用 A2；未指定时使用数据集公共向量 |
| GZSL | 仅 seen 活动，并排除被抽作测试的 seen 窗口 | seen 与 unseen 活动 | seen ∪ unseen | 可复用同一 activity split 的 A2；支持 seen-bias 校准 |
| 跨 subject | train subjects 的全部活动 | 完全未参与训练的 test subjects | 全部活动 | 当前固定使用数据集配置中的公共 A1；不接受 split-specific A2 root |

ZSL/GZSL 的划分轴是“活动类别”。同一个 subject 可以同时在 seen 训练窗口和 unseen
测试窗口中出现。跨 subject 的划分轴是“人员”，活动标签空间没有 seen/unseen 划分。

## 当前实现边界

1. `planner1.py` 使用完整 A1 活动原型、真实 seen/unseen 名单和语义 D_kNN/近邻，
   不再筛选 issue cards。它利用目标类别语义，但不读取 unseen 记录或测试预测。
   当前策略及重跑命令见 [完整 A1 细化说明](FULL_A1_REFINEMENT.md)。
2. `a2_attribute_plan.feature_mapping.json` 已经给出每个属性绑定后的精确特征列和索引，
   但 `pipeline/runner.py` 与 `pipeline/training.py` 当前仍把 `DatasetTable.feature_columns`
   的全部列传给每个属性模型。因此，绑定目前是可观测性审核和可追溯产物，还不是训练时
   的逐属性 feature routing。
3. `pipeline/runner.py` 当前拒绝在 `cross_subject` 场景传入 `attribute_vectors_root`，而
   `planner1.py` 也拒绝仅凭 activity-split CSV 执行 cross-subject 细化。因此跨 subject
   使用数据集配置的公共属性向量，当前没有 subject-split-specific A2。
4. 属性预测器由 `--backend` 决定。普通后端独立预测各属性；多任务后端联合预测；
   classifier-chain 后端按指定顺序预测。`autogluon_multilabel` 训练后序模型时使用前序
   真实属性，推理时使用前序预测属性。

