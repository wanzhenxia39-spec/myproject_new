# 属性细化前的 A1 GAP / D_kNN 分析

`src/zshar/semantics/gap.py` 默认使用 `normalized_ordinal` 原型距离，并保留 `mad` 和 `euclidean` 兼容选项。
新增的 `src/zshar/semantics/gap_analysis.py` 负责读取最新 A1、验证活动及属性集合、加载划分，以及生成 planner 可直接读取的 CSV。
`zshar gap` 命令也调用该入口。

## 输入

- 默认读取 `configs/datasets/<dataset>.json` 的 `attributes` 路径；当前五个主数据集均指向 `configs/attributes/activity_attribute_vectors_initial_<dataset>_v1_5.json`。
- 可通过 `--a1-json` 显式指定本次使用的 A1 向量文件。每次运行重新读取文件，不按修改时间猜测文件、不使用旧十维向量作为替代。
- 活动必须与数据集配置完全一致，属性字段及 Low/Medium/High 取值必须符合 `configs/attributes/initial_attributes.json`。
- 默认使用 `split/zero_shot_split/` 中对应 seed 的划分文件；当前五个数据集的 seed 42 文件各包含 20 个划分。可用 `--split-manifest` 覆盖。
- 默认划分文件不存在时，沿用原实现的活动组合生成逻辑；显式指定的划分文件不存在则报错。
- 不读取窗口特征、原始 IMU 或任何测试样本。

`initial_attributes.json` 是属性定义，不是活动向量，不能作为 `--a1-json` 输入。

## 计算规则

默认距离定义为 `d_A(i,j) = (1/K_A) * sum_k(abs(a_i,k-a_j,k)/2)`，L/M/H 编码为 0/1/2。
内部采用等价的 `Low=0.0`、`Medium=0.5`、`High=1.0` 表示；按当前属性数 K_A 取平均，
距离范围为 [0,1]，不引入分类器可靠性权重。该公式与原等权 `mad` 数值等价，
不会因 A2 增加属性而机械累加距离。可用 `--distance-metric mad` 保留旧接口。

对于每个 unseen 活动，先计算它与所有 seen 活动的距离，取最近 `k` 个距离的平均，作为该 unseen 活动的 D_kNN；再对所有 unseen 活动的 D_kNN 取平均，得到划分总分。
默认 `k=2`；seen 活动少于 k 时使用全部 seen 活动。可用 `--distance-metric euclidean` 采用原代码支持的加权均方差开根号距离。

原实现的 `U_NN`、`U_Conf`、`JointGap_DkNN_UConf` 也保留；`difficulty` 仍按 JointGap 分组，不是按 D_kNN 单独分组。只有一个 unseen 活动时 U_NN 无定义，D_kNN 仍可计算，difficulty 标记为 Unranked。

## 运行

先生成活动级 A1 向量，然后执行：

```bash
cd /home/wanzhen/myproject_new
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python \
  -m zshar.semantics.gap_analysis \
  --dataset pamap2 --top-k 2 --distance-metric normalized_ordinal
```

默认输出 `runtime/gap/pamap2/a1_gap_seed42.csv` 及同名 `.meta.json`。
运行根目录可通过项目现有 `ZSHAR_RUNTIME_ROOT` 环境变量调整。

显式指定向量文件和输出位置，也可使用统一 CLI：

```bash
PYTHONPATH=src /home/wanzhen/miniconda3/envs/zara/bin/python -m zshar.cli gap \
  --dataset pamap2 \
  --a1-json configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json \
  --output runtime/gap/pamap2/a1_gap_seed42.csv
```

将 `pamap2` 换成 `mhealth`、`dsads`、`uschad`、`wisdm` 即可用于相应数据集。

## 输出与属性细化衔接

CSV 包含：

| 字段 | 含义 |
|---|---|
| `split_index` | 保留原划分编号，供 planner 选择 |
| `seen_classes`, `unseen_classes` | 以竖线分隔的活动名称，符合现有 planner 读取格式 |
| `seen_key`, `unseen_key` | 排序后的活动集合键 |
| `D_kNN` | 划分总分 |
| `D_kNN_per_unseen_json` | 每个 unseen 活动的 D_kNN，标准 JSON |
| `U_NN` | 所有 unseen 活动到最近其他 unseen 活动的平均距离 |
| `U_NN_per_unseen_json` | 每个 unseen 活动到最近其他 unseen 活动的距离，标准 JSON |
| `nearest_seen_topk_json` | 每个 unseen 活动最近的 seen 活动及距离，标准 JSON |
| `top_k`, `effective_k`, `distance_metric` | 请求的 k、实际 k 和距离类型 |
| `a1_json`, `a1_sha256` | 本次使用的向量路径和内容哈希 |

原有结构字段也保留，并序列化为标准 JSON。`.meta.json` 保存向量快照、实际划分、属性顺序及计算参数。

后续 planner 的 `--a1_json` 应使用本次计算 GAP 的同一份向量，`--dknn_csv` 指向上面的 CSV；`--split_index` 使用 CSV 中的实际编号。不要将 A1 更新后的向量与更新前的 GAP 混用。
现有 `resolve_split_from_gap_csv()` 可以直接读取这些字段，无需手动转换。

现有五个主要数据集均已按 seed=42 的 20 个 split 重算 A1 指标，输出为
`runtime/gap/<dataset>/a1_gap_normalized_ordinal_seed42.csv` 及对应元数据。
PAMAP2、MHEALTH、WISDM 的 A1/A2 重算对比位于
`runtime/gap/normalized_ordinal_seed42/<dataset>/`。复现命令及生成策略说明见
[完整 A1 细化说明](FULL_A1_REFINEMENT.md)。
