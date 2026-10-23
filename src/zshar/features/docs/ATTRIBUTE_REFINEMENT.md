# 逐 split 属性细化与特征绑定

本文主要描述保留的 `planner.py`。当前批处理使用 `planner1.py` 的完整 A1 策略，
输入、操作、数量范围及命令以 [完整 A1 细化说明](FULL_A1_REFINEMENT.md) 为准。

旧项目 `src/zshar/semantics/planner.py` 的主体是通用流程，通过数据集名称、传感器元信息、活动向量和 GAP CSV 区分任务。旧提示词的属性定义和细化规则带有历史假设；本次在新项目对应模块中替换了这些规则，原项目文件保留。

新入口为 `src/zshar/semantics/planner.py`，不涉及属性分类器训练。五个已注册数据集 PAMAP2、MHEALTH、DSADS、USC-HAD、WISDM 可直接使用配置中的路径；其他已配置特征前缀的数据集可显式提供 `--feature-names-npy`、`--metadata-json`、A1 和 GAP CSV。新增数据集仍需配置其实际前缀及固定组模板，不能假定任意数据集自动可用。

## 两阶段流程

1. **概念细化**：每个 split 独立读取初始向量、D_kNN、逐 unseen 距离及近邻 seen 类。优先分析覆盖较差的 unseen 类与其他 unseen 类的物理差异，同时检查 seen 类能否提供可迁移证据。输出属性定义、三级含义、条件判断和 `keep/split/merge/drop` 操作。
2. **重新绑定特征组**：第二次 LLM 调用只接收已确定的概念、固定特征体系和实际可用列。每个 A2 属性，包括保留的属性，都必须重新选择至少一个主组，解释所有主/辅助组的物理支持，必要时指定特征侧筛选条件。第二阶段响应不能修改概念。
3. **校验和保存**：校验操作来源覆盖、属性名、等级、语义重复的基本结构、组名、证据字段以及实际可用列；保存最终方案和每个属性的精确列名、列索引、结构标签。

第一阶段删除了原提示词中的 `Initial attribute set A1` 定义块及旧 A1 组绑定，但保留**来源属性名和活动级初始向量**：它们仍是 D_kNN 的解释依据和操作追踪输入。原 Section 4 的去冗余要求合并进 Section 3。没有“至少新增 4 个属性”等配额；无充分理由时允许全部保留。

`keep` 为一来源对应同名输出；`split` 为一来源对应至少两个输出；`merge` 为至少两个来源对应一个输出；`drop` 为一来源对应空输出。每个输入、输出必须被追踪且只能出现一次。合并明确记录多个来源。拆分和合并须给出目标 unseen、对比 unseen（有其他 unseen 时）及 seen 锚点。当前有意只提供这四种操作，没有独立新增或一对一改名操作。

属性表达抽象运动物理概念。名称中的常见部位、传感器、轴和组编号会被拒绝；属性语义字段之外的额外字段也会被拒绝。提示词要求同时审查定义及等级中的隐含部位限制。**结构校验不能完整理解自然语言**：同义概念、物理支持是否充分及拆分的真实价值仍依赖 LLM 的语义审查和后续 seen 数据验证。相同定义会被直接拒绝，统计相关或相同等级模式不会被当作语义重复自动删除。

D_kNN 是细化的诊断及设计目标，不能通过删除有用差异或压平向量来机械降低。本阶段不生成或测量 A2 原型，故不声称优化已成功；A2 向量生成后仍需比较 D_kNN、unseen 区分性及 seen 可预测性。带有 `a1_sha256` 的 GAP CSV 会校验当前 A1 内容，旧 CSV 缺少哈希时会在元数据中记录未验证。

## 特征侧筛选

每个绑定包含：

```json
{
  "attribute": "cadence",
  "primary_groups": ["G6"],
  "auxiliary_groups": [],
  "group_evidence": {
    "G6": "主频和自相关峰间隔提供动作重复速率证据；需注意谐波和非周期动作。"
  },
  "feature_selectors": {"positions": [], "modalities": [], "axes": []}
}
```

空列表表示不限；`positions` 使用真实列名前缀，如 `hand`，不是物理部位描述文本；`modalities` 为 `acc/gyro`；`axes` 为 `x/y/z`。三类条件组合执行，保留原始列顺序。每个选中组在筛选后都必须至少剩余一列，不能静默退回全部特征。

`features/catalog.py` 保留每列的组、位置前缀、实际身体部位、模态、轴和统计量标签。多轴/多模态派生量只有在其**所有依赖标签**均满足筛选条件时才保留。例如 EVA、CAGH 和幅值依赖三轴，acc_gyro_ratio 依赖 acc 和 gyro；仅选择 x 轴时不会保留三轴派生量。AVG、AVH、ARATG 的轴依赖按当前 `PhysicalFeatureConfig` 默认配置解释（重力轴索引 0，水平轴索引 1、2），这不意味着任意佩戴条件下 x 轴都等于真实重力方向。若更改抽取配置，需同步更新标签逻辑。

默认读取数据集配置的 `feature_names` 文件，**不读取 IMU 样本或 unseen 数据统计**。MHEALTH 已在 2026-09-11 从全部 1104 个既有窗口重抽为 434 列，G1–G14 均有实际列；旧 386 列文件已备份。任何数据集若再次出现缺失组，该组不会成为合法绑定，不能通过模板推测特征已经存在。`--templates-only` 仅允许 dry-run 提示词预览，不能据此生成已校验的正式映射。

`completion.py` 只将 `name/definition/levels` 传给向量生成 LLM，不传特征组、筛选条件和 split 的设计理由，避免重新把传感器条件混入活动语义。

如果一次局部 replan 后仍有属性被绑定审核拒绝，整个 split 回退到固定 A1 九属性及原始组映射。此时 `completion.py` 校验方案中记录的 A1 路径和 SHA-256，直接复制原 A1 活动属性向量并生成本地检查报告，不构造向量提示词，也不调用 LLM。批处理会把历史上通过 LLM 重生成的 fallback 向量视为未完成，并在续跑时替换为该 A1 文件的精确副本。

## 运行方式

必须指定新项目的包路径，避免加载环境中旧项目的同名包：

```bash
cd /home/wanzhen/myproject_new
export PYTHONPATH="$PWD/src"
```

下面 A1、CSV 路径为占位示例，请换成已经生成的文件。先生成所有 CSV split 的第一阶段提示词，不调用 API：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -m zshar.semantics.planner \
  --dataset-name pamap2 \
  --a1-json /path/to/current_a1_vectors.json \
  --dknn-csv /path/to/current_a1_gap.csv \
  --all-splits --output-root runtime/refinement/pamap2 \
  --dry-run
```

去掉 `--dry-run` 即会逐 split 运行两次 LLM 调用，沿用原 planner 的 Gemini transport。需要 `google-genai`，环境变量 `LLM_API_KEY` 或 `GEMINI_API_KEY`，可选 `LLM_BASE_URL`、`LLM_TRUST_ENV`。本次没有切换 planner 的网络接口实现。

单 split 使用 `--split-index N --output-json runtime/refinement/pamap2/split_N/a2_attribute_plan.json` 替换 `--all-splits --output-root ...`。也可使用 `--unseen` 精确选择 CSV 行，不能覆盖行中的 seen/unseen 定义。下划线形式的参数名仍可使用；旧的 `--min_new_attributes/--max_new_attributes` 配额参数已移除，可用 `--min-attributes/--max-attributes` 限定最终数量，默认 1–25。

中断恢复、离线校验：

- `--concepts-json /path/to/saved_concepts.json --dry-run`：校验已保存概念，只生成第二阶段提示词。
- `--concepts-json ...`：复用概念，只调用第二阶段。
- `--concepts-json ... --bindings-json /path/to/saved_bindings.json`：完全离线重放、校验和导出。
- 批量重放使用 `--responses-root DIR`，其中每个 `split_N/` 包含 `concepts.json` 和 `bindings.json`。缺失、无效响应会报错，不自动发起替代调用。

可以直接复用同一输出目录中的 `.concepts.json`、第一阶段或第二阶段原始响应文件；重放时保留这些输入文件，不覆盖它们。最终方案、映射和提示词路径不能覆盖输入。运行失败时保留已经写出的提示词和响应便于检查；不会把无效响应保存为最终方案。批量遇到错误会停止，已完成 split 文件仍在，可按单 split 入口继续。

每个 split 保存以下产物（前缀取决于 `--output-json`）：

| 文件 | 用途 |
|---|---|
| `a2_attribute_plan.prompt.txt`、`.raw_response.txt` | 概念阶段输入和原始响应 |
| `a2_attribute_plan.concepts.json` | 通过校验的概念方案及操作记录 |
| `a2_attribute_plan.binding.prompt.txt`、`.binding.raw_response.txt` | 绑定阶段输入和原始响应 |
| `a2_attribute_plan.json` | 最终 A2 定义、组绑定及来源审计 |
| `a2_attribute_plan.feature_mapping.json` | 每属性精确列名/索引、所有原始列结构标签 |
| 批量输出根目录 `manifest.json` | CSV 各 split 的输出索引 |

方案和映射都记录 dataset、split、seen/unseen、A1/GAP 哈希、特征列顺序哈希和分组配置哈希。该映射已可供后续训练接入；**本次没有修改独立属性分类器或 classifier chain 的训练、推理、重载逻辑**。

## 验证

```bash
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python -m pytest
```

新增测试覆盖四种操作及来源错误、无变化方案、非法传感器属性名、语义重复基本检查、绑定完整性、第二阶段语义冻结、各数据集配置的组映射、缺失实际列、多轴/多模态派生量筛选、GAP split/hash 校验、两次模拟调用以及批量离线重放。测试不调用真实 LLM，也不证明概念质量或分类性能。
