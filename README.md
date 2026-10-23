export KAMI_API_KEY='YOUR_API_KEY'
# 特征抽取与九维初始属性生成

初始属性向量生成后的 D_kNN 计算及属性细化输入衔接见 [A1 GAP 使用说明](docs/A1_GAP.md)。
当前完整 A1 + 真实 seen/unseen 属性细化及 MHEALTH 重跑见 [完整 A1 细化说明](docs/FULL_A1_REFINEMENT.md)。
保留的 `planner.py` 入口及特征筛选说明见 [原两阶段细化说明](docs/ATTRIBUTE_REFINEMENT.md)。
MHEALTH 九维 A1 的 AutoGluon 默认集成、逐属性特征绑定及 GPU 0 后台命令见 [AutoGluon 绑定特征实验](docs/MHEALTH_A1_AUTOGLUON_BOUND.md)。

本目录包含从原始 IMU 数据预处理、分窗、特征抽取，到基于统一九维定义生成活动级 A1 属性原型的代码。

## 代码位置

| 阶段 | 位置 |
|---|---|
| 单窗口统计和物理特征抽取 | `src/zshar/features/all_features.py` |
| 数据集预处理、分窗、批量抽取 | `src/zshar/preprocessing/preprocess_*.py` |
| 从既有 MHEALTH 全部窗口安全重抽完整特征 | `src/zshar/preprocessing/reextract_mhealth_features.py` |
| 五个主要数据集的统一预处理入口 | `src/zshar/preprocessing/cli.py` |
| G1–G14 模板、实际传感器前缀及模态 | `configs/features/feature_groups.json` |
| 特征模板展开、实际列名分组、显式绑定后的属性列选择 | `src/zshar/features/groups.py` |
| 九维 A1 属性名、定义和三级含义 | `configs/attributes/initial_attributes.json` |
| A1 各数据集重点对比活动及名称说明 | `configs/attributes/initial_vector_prompt_context.json` |
| A1 提示词、LLM 调用、向量校验和保存 | `src/zshar/semantics/initial_vectors.py` |
| 逐 split 概念细化、第二阶段特征绑定 | `src/zshar/semantics/planner1.py` |
| keep/split/merge/drop/replace/add 来源及绑定校验 | `src/zshar/semantics/refinement.py` |
| 结构化特征标签、属性侧组与位置/模态/轴筛选 | `src/zshar/features/catalog.py` |
| 按抽象属性定义生成 A2 活动级向量 | `src/zshar/semantics/completion.py` |
| 当前数据集活动名称和 A1 输出路径 | `configs/datasets/*.json` |
| UCI-HAR、Shoaib 的历史活动名称 | `configs/activities/*.json` |

预处理代码来自旧项目的 `legacy/LLMbasedZeroshotHARAgent/version1/`。
九个 `preprocess_*.py` 以及 `HHAR_preprocess.py` 均已迁移，统一导入新项目的特征抽取模块。
PAMAP2、MHEALTH、DSADS、USC-HAD、WISDM 提供统一命令入口；UCI-HAR、Shoaib 和 HHAR 保留各自的历史函数接口。
旧 `attribute_vector_generation_CoT_*.py` 的通用内部推理框架、数据集上下文和易混活动对比要求已合并为共享 A1 模块；新项目不再维护各数据集重复启动脚本。`pamap` 在新项目统一使用数据集键 `pamap2`。

## 运行环境

在新项目根目录运行，显式指定 `PYTHONPATH`，避免环境里旧项目的同名 `zshar` 包被加载：

```bash
cd /home/wanzhen/myproject_new
export PYTHONPATH="$PWD/src"
```

以下命令使用已验证的 Python 环境：`/home/wanzhen/miniconda3/envs/zara/bin/python`。
项目依赖中已补充特征抽取所需的 SciPy。

## 预处理与特征抽取

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -m zshar.preprocessing.cli \
  --dataset pamap2 \
  --data-path /path/to/PAMAP2_Dataset/Protocol \
  --save-path runtime/windows/pamap2 \
  --target-sr 20 --seq-len 120
```

`--dataset` 支持 `pamap2`、`mhealth`、`dsads`、`uschad`、`wisdm`。
也可运行 `python -m zshar.preprocessing.preprocess_pamap2` 并传入同样的路径参数。
DSADS 模块文件名保留为 `preprocess_DSADS.py`。

输入目录遵循各数据集原始格式：MHEALTH 的 `.log`、DSADS 的 `aXX/pX/sXX.txt`、USC-HAD 的 `SubjectX/aXtY.mat`、WISDM 的 `raw/{phone,watch}/{accel,gyro}/`。

默认输出：

```text
data_20_120.npy                # (窗口数, 120, 传感器通道数)
label_20_120.npy               # (窗口数, 120, 2)：活动 ID、用户 ID
all_features_20_120.npy        # (窗口数, 特征数)
all_features_name_20_120.npy   # 与特征矩阵列顺序一致的列名
```

可用 `--version` 指定输出后缀。若改用新的输出目录，需要同步修改 `configs/datasets/<dataset>.json` 中的 `files` 路径，训练框架才会读取新产物。
UCI/Shoaib/HHAR 的历史脚本具有各自的输入布局及标签约定，不默认接入上述五数据集的训练配置。

## `{prefix}` 如何替换

`feature_groups.json` 的 `feature_templates` 保留可替换模板，例如 `{prefix}_acc_x_mean`。
同文件的 `dataset_sensors` 明确列出实际代码前缀、物理部位说明、已有传感器模态。
物理部位说明不直接拼到特征名中；前缀必须与预处理输出一致。
`{prefix}` 在 `expand_feature_groups(dataset)` 执行时，按 `dataset_sensors` 的键替换。它不是任意自然语言部位名；替换结果必须与预处理传给抽取器的 sensor dict 前缀完全一致。抽取器本身不会读取或替换 JSON 模板。

| 数据集 | 实际特征前缀 | 完整输出列数 |
|---|---|---:|
| PAMAP2 | `hand`, `chest`, `ankle` | 534 |
| MHEALTH | `hand`（右下臂）, `chest`, `ankle`（左脚踝） | 434 |
| DSADS | `torso`, `right_arm`, `left_arm`, `right_leg`, `left_leg` | 890 |
| USC-HAD | `hip`（右前髋部） | 178 |
| WISDM | `phone`, `watch` | 356 |

MHEALTH 的 `chest` 只有 `acc`。展开时不会生成 `chest_gyro_*`，也不会生成当前抽取器要求 acc+gyro 配对才能输出的胸部物理特征。
模板的大小写与抽取器输出严格一致，例如 `hand_ACC_X_DF`、`hand_acc_x_fft_dom_freq` 是两个不同特征。

不读取数据即可导出某个数据集完整的具体特征名：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -m zshar.features.groups \
  --dataset mhealth --templates-only --output runtime/mhealth_feature_groups.json
```

省略 `--templates-only` 则检查数据集配置指向的实际特征列并输出 G1–G14
分组。只有通过 `--attribute-bindings` 显式传入细化后绑定方案时，才会进一步
生成每个属性的支持列及索引；不会从 A1 ontology 推断映射。
Python 接口支持自定义前缀：

```python
from zshar.features.groups import expand_feature_groups

groups = expand_feature_groups(
    "mhealth", prefix_overrides={"hand": "right_lower_arm"}
)
```

这个参数只替换展开结果的前缀；如果需要更改实际输出列名，还必须同步更改预处理的 sensor dict 键和 `channel_names`，并重新生成特征文件。
自动化测试使用各数据集配置的传感器组合调用真实 `extract_features()`：PAMAP2 534、MHEALTH 434、DSADS 890、USC-HAD 178、WISDM 356、UCI-HAR 178、Shoaib 178、HHAR 178 列均与 G1–G14 展开结果逐列一致，没有未分组抽取特征或不会被抽取的多余模板。

## 九维初始属性 A1

`initial_attributes.json` 中每个属性只保留 `name`、`definition` 和
`levels`，不保存初始属性与特征组的映射。属性定义及 Low/Medium/High
含义集中保存在该配置中。
`posture_inclination` 按“平均倾斜程度”定义；姿态是否稳定由 `posture_variability` 表达。
新 A1 不包含旧的 `body_coordination`、`motion_type`、`vertical_motion`、`spatial_variability`。
三级含义保留明确约定：没有周期性动作时 cadence 为 Low，近乎静止且稳定的信号可对应 High smoothness/regularity。

生成器保留旧版 CoT 提示词的通用步骤：解释属性、分析活动生物力学、比较数据集内易混活动、赋值和一致性审查。各数据集的活动及重点对比项由配置注入。A1 LLM 不接收 G1–G14、属性—特征组映射、具体特征模板、传感器清单、原始 metadata、传感器样本、特征值或 seen/unseen 划分；A1 只描述活动级抽象物理语义。
A1 提示词只接收 `name/definition/levels`，不接收任何特征组对应关系。
最终可执行映射在 A2 细化完成后重新生成。
生成的是活动级语义原型；窗口级属性预测仍由后续分类器完成。

先生成提示词检查：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -m zshar.semantics.initial_vectors \
  --dataset pamap2 --dry-run
```

实际调用读取环境变量 `LLM_API_KEY`；Gemini 格式也兼容 `GEMINI_API_KEY`、
`ZZZ_API_KEY`，OpenAI-compatible 格式也兼容 `KAMI_API_KEY`、`NEW_API_KEY`、
`OPENAI_API_KEY`、`ZZZ_API_KEY`。
支持项目现有 transport 的 Gemini 和 OpenAI 兼容格式：

```bash
# 先在环境中配置 LLM_API_KEY；需要代理时配置 LLM_BASE_URL。
/home/wanzhen/miniconda3/envs/zara/bin/python -m zshar.semantics.initial_vectors \
  --dataset pamap2 --api-format gemini --model gemini-2.5-flash

# 对应的数据集脚本入口：
/home/wanzhen/miniconda3/envs/zara/bin/python \
  scripts/attribute_vector_generation/attribute_vector_generation_CoT_mhealth.py --dry-run
```

可用 `--api-format openai --model <model>` 切换兼容接口。
`--response-json <file>` 可以读取已有 LLM 响应，执行相同的严格校验与保存，不发起 API 请求。

Kami 的 `/v1/chat/completions` 是 OpenAI-compatible 接口。生成器会读取
`LLM_API_KEY`、`KAMI_API_KEY` 或 `NEW_API_KEY`，不需要把密钥写入命令或代码。
建议先把实际调用将使用的完整 system/user messages 保存到 runtime 目录检查。
程序同时生成与 API 一致的 `.prompt.json` 和适合人工阅读的 `.prompt.txt`：

```bash
cd /home/wanzhen/myproject_new
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python \
  -m zshar.semantics.initial_vectors \
  --dataset mhealth \
  --api-format openai \
  --base-url https://www.kamiapi.top/v1 \
  --model gpt-5.5 \
  --prompt-out runtime/a1_preview/mhealth.prompt.json \
  --dry-run
```

确认提示词后调用 Kami；推荐在当前 shell 中交互式读入密钥，避免进入 shell history：

```bash
read -s -p "Kami API key: " NEW_API_KEY; echo
export NEW_API_KEY

PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python \
  -m zshar.semantics.initial_vectors \
  --dataset mhealth \
  --api-format openai \
  --base-url https://www.kamiapi.top/v1 \
  --model gpt-5.5 \
  --temperature 0 \
  --timeout 300

unset NEW_API_KEY
```

真实调用也会在发出 HTTP 请求之前先保存同一份 `.prompt.json` 和 `.prompt.txt`。
JSON 中的 `content` 必须按 JSON 标准转义换行；TXT 会将 system/user 分段并把
user 中的嵌套 JSON 展开缩进。接口响应原文、
严格校验后的九维向量和检查报告分别保存为 `.raw_response.txt`、`.json` 和
`.report.json`。`--base-url` 也接受完整地址
`https://www.kamiapi.top/v1/chat/completions`。

默认保存到 `configs/attributes/activity_attribute_vectors_initial_<dataset>.json`，与现有五个数据集配置一致。可用 `--output` 覆盖。
同时保存 `.prompt.json`、`.raw_response.txt` 和 `.report.json`。
输出必须精确包含所有目标活动、全部九个属性，且取值只能是 Low/Medium/High。
报告标记常量属性和重复完整向量，但不自动删除属性或强行改变语义取值。

## 迁移修复与验证

- 保持单窗口 `all_features.py` 算法和数值输出不变，修正所有迁移脚本的导入路径。
- 修复 MHEALTH 批量抽取的 `channel_names` 与 `hand/chest/ankle` 字典键不一致，避免漏掉 48 列物理特征。
- MHEALTH 标签固定为原始活动 ID 减 1，WISDM 标签固定为 `ABCDEFGHIJKLMOPQRS` 的索引，避免缺少活动或改变出现顺序时与数据集配置错位。
- Shoaib 第一阶段原来在 import 时执行预处理，已移入 `__main__` 保护。
- 五个主要数据集均使用合成原始格式文件验证读入、分窗、保存、活动编号和 G1–G14 完整覆盖；A1 测试覆盖七个数据集、旧 schema 拒绝、非法等级拒绝、dry-run、离线响应和模拟 API 调用。

```bash
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python -m pytest
```

2026-09-11 已从 `/home/wanzhen/windows/mhealth_windows/data_20_120.npy` 的全部 1104 个窗口重新抽取 MHEALTH 特征。当前矩阵为 `(1104, 434)`，列名为 `(434,)`，G1–G14 全部非空且无缺失模板；旧 386 列文件保存在同目录的 `*.incomplete_386_backup.npy`。
如需从既有窗口再次安全重抽，可运行：

```bash
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python \
  -m zshar.preprocessing.reextract_mhealth_features
```

脚本校验窗口/标签数量、完整 434 列 schema、有限值和 G1–G14 覆盖，完成前写入 partial 文件，成功后原子替换；中断后加 `--resume` 继续。
其余四个主要数据集现有特征名也与模板完全匹配。
本次没有运行全量原始数据处理，也没有向外部 LLM 请求生成实际 A1 向量；代码验证使用合成数据和模拟响应。
