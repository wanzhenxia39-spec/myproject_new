# 属性原型分析：系统盘代码与结果

正式代码目录：`/home/wanzhen/attribute_prototype_analysis/`。
它是真实系统盘目录，不是指向数据盘的软链接。此前项目内的这几份代码已迁至此处。
原项目 `/home/wanzhen/myproject_new` 实际指向 `/data/wanzhen/projects/myproject_new`，
因此不适合作为本次要求的系统盘代码目录。

默认原型输入保留在数据盘项目的
`configs/attributes/activity_attribute_vectors_initial_<dataset>_v1_5.json`。
PAMAP2、MHEALTH、USC-HAD、WISDM、DSADS 分别包含 12、12、12、18、19 个活动，
全部使用细化前最新版九维初始属性集，不涉及 split。

## 独立 Attribute Discrimination Coverage

覆盖率公式已独立到 `attribute_discrimination_coverage.py`，仅依赖 Python 标准库。
支持任意属性数、离散等级、JSON/CSV 及多方法/多数据集矩阵，不需要活动距离或模型文件。
综合分析入口也复用这一份公式。详见 [ATTRIBUTE_COVERAGE.md](ATTRIBUTE_COVERAGE.md)。

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/attribute_discrimination_coverage.py \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/attribute_coverage_a1_v1_5
```

迁移到其他属性方案使用 `--vectors`（单数据集）或 `--manifest`（多数据集/多方法）。

## 独立严格 APCR

**严格 APCR 的唯一计算实现为 `exact_apcr.py`**。该文件可独立运行或复制使用，
仅依赖 Python 标准库，不依赖综合分析脚本、numpy、pandas、matplotlib 或项目环境。
综合分析入口也调用这里的 `compute_exact_apcr`，不再保留另一份公式实现。

定义：令 C 为活动类别数，K 为属性数，

`APCR_exact = count(i<j: 对每个 k 均有 p_ik == p_jk) / [C(C-1)/2]`。

只统计不同活动的无序类别对；所有坐标同时相同才算冲突；不使用容差。
原型必须包含相同的非空属性集合，至少有两个活动。
Low/Medium/High 默认对应 0/1/2，其他类别标签精确比较；缺失、重复键、NaN 等会报错。
数值采用 Decimal 保持精确相等判断，不进行浮点距离近似或 min-max 归一化。

计算默认五数据集：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/exact_apcr.py \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/exact_apcr_a1_v1_5
```

输出：

```text
<output>/
  dataset_summary.csv      # 每数据集严格 APCR、冲突对数、总类别对数
  method_summary.csv       # 每方法的跨数据集等权平均和类别对数量加权平均
  run_metadata.json        # 输入路径、输入/代码哈希、活动/属性、公式
  <method>/<dataset>/
    summary.json
    exact_collision_pairs.csv
```

这里的严格 APCR **不是**单属性同值率 `1-R_k`，也不是九属性同值率的平均。
默认五数据集严格 APCR 均为 0，等权平均也为 0。

计算其他方法，例如自备 IMUZero 原型 JSON 或 CSV：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/exact_apcr.py \
  --vectors /data/wanzhen/experiments/prototype_inputs/imuzero_pamap2.json \
  --dataset pamap2 --method imuzero \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/imuzero_exact_apcr
```

JSON 格式为 `{活动名称: {属性名称: 等级}}`。CSV 的首列为 `activity`，其余列为属性。
自定义类别标签无需有序；严格 APCR 只检查相等。

批量比较多方法、多数据集可用 `--manifest /data/.../methods.json`：

```json
[
  {"method": "ours_a1", "dataset": "pamap2", "vectors": "ours_pamap2.json"},
  {"method": "imuzero", "dataset": "pamap2", "vectors": "imuzero_pamap2.csv"}
]
```

相对输入路径基于清单目录解析。每个方法/数据集只能出现一次；不同方法可有不同
属性数，但同一数据集的活动集合必须相同。条目可使用 `level_map` 显式定义标签对应关系，
`level_map: {}` 表示按原标签比较。严格 APCR 无需 `value_range`。

Python API（在代码目录下运行或将该目录加入导入路径）：

```python
from exact_apcr import read_vectors, compute_exact_apcr

result = compute_exact_apcr(read_vectors('/data/wanzhen/experiments/prototype_inputs/prototypes.json'))
print(result['exact_apcr'])
print(result['collision_pairs'])
```

## 独立活动原型两两距离热力图

`prototype_distance_heatmap.py` 为单文件入口。支持 JSON/CSV、可选距离、可选验证准确率权重、
五数据集批量运行和 PDF/SVG/PNG 导出，不依赖其他本地 Python 文件。

当前默认 `--metric ordinal_probability`，表示最终匹配距离在确定性原型上的单点概率分布特例。
**等权时数值与原先 ordinal L1 完全相同**。这不是模型对真实样本预测的不确定概率距离；
默认不读取模型结果，也不自动引入某个 split 的验证准确率。
详细公式、方法选择依据与局限见 [DISTANCE_METRICS.md](DISTANCE_METRICS.md)。

一次生成五数据集主图，并生成归一化欧氏 RMS 对照：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/prototype_distance_heatmap.py \
  --datasets pamap2 mhealth uschad wisdm dsads \
  --metric ordinal_probability --comparison-metrics euclidean \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/distance_heatmaps
```

主图位于 `results/distance_heatmaps/<dataset>/`，欧氏对照位于
`results/distance_heatmaps/comparisons/euclidean/<dataset>/`。
各目录含 `distance_matrix.csv`、`distance_pairs.csv`、`normalized_prototypes.csv`、
`prototype_distance_heatmap.{pdf,svg,png}` 和 `distance_run_metadata.json`。
根目录 `distance_summary.csv` 汇总两种度量的结果，`batch_run_metadata.json` 记录本次批量口径。
热力图色阶全部固定为 `[0,1]`。旧图和旧代码已保存在 `_archive/` 下带时间戳的目录。

单个数据集示例：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/prototype_distance_heatmap.py \
  --dataset mhealth --metric ordinal_probability
```

其他方法使用 `--vectors /实际路径/prototypes.json` 或 CSV；其他有序等级用 `--level-map-json`
和 `--value-range` 显式定义。原型依旧来自最新九维 A1 v1.5，使用全部活动，不涉及 split。
可选距离为 `ordinal_probability`、`ordinal_l1`、`manhattan`、`euclidean`、`cosine`、`hamming`。

如需与某个已训练实验的属性权重一致，单数据集模式加 `--weights-json /实际路径/validation_report.json`。
文件可为完整 `属性名 -> 权重` 字典，或项目保存的 `per_attribute -> 属性名 -> accuracy` 报告。
所有属性必须显式提供。不同数据集或 split 的验证权重不能隐式混用；批量模式拒绝单个权重文件。

Python API：

```python
from prototype_distance_heatmap import load_prototypes, pairwise_prototype_distances

table = load_prototypes('/data/wanzhen/projects/myproject_new/configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json')
distance_matrix = pairwise_prototype_distances(table, metric='ordinal_probability')
```

`pairwise_ordinal_distances(table)` 仍保留旧的等权 ordinal L1 接口，综合覆盖率分析继续使用该接口。

## 距离、覆盖率及论文插图

综合入口为 `analyze_attribute_prototypes.py`，对应计算和绘图函数在 `discriminability.py`。
该入口同时调用独立 APCR 文件，便于一次性导出全套分析。

- 距离使用附件图 1：`d(i,j)=sum_k |p_ik-p_jk|/(2K)`，L/M/H 固定编码 0/1/2。
- 覆盖率使用附件图 2：`R_k=count(i<j:p_ik!=p_jk)/[C(C-1)/2]`，不按等级差加权。
- 单属性同值率为 `1-R_k`，与完整原型严格 APCR 分开保存。
- 默认仅绘制 PAMAP2 活动距离图，以及九属性 × 五数据集覆盖率矩阵。
- 两张图均使用 Blues、固定 `[0,1]` 色阶，输出矢量 PDF、SVG 和 600 dpi PNG。
- 图上显示两位小数，CSV 保留完整精度；默认不添加标题，以便使用论文图注。

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/analyze_attribute_prototypes.py \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/a1_v1_5_paper
```

输出路径保持为：

```text
/home/wanzhen/attribute_prototype_analysis/results/a1_v1_5_paper/
  dataset_summary.csv
  method_summary.csv
  all_attribute_metrics.csv
  run_metadata.json
  initial_a1_v1_5/
    attribute_coverage_matrix.csv
    attribute_coverage_heatmap.{pdf,svg,png}
    single_attribute_collision_matrix.csv
    attribute_dataset_means.csv
    pamap2/
      prototype_distance_heatmap.{pdf,svg,png}
      distance_matrix.csv
      attribute_metrics.csv
      activity_pairs.csv
      exact_collision_pairs.csv
      normalized_prototypes.csv
      summary.json
    mhealth/ ...  # 各数据集同样导出 CSV/JSON
```

其他数据集距离图可用 `--distance-datasets pamap2 mhealth uschad wisdm dsads`。
综合入口支持上述 `--manifest` 格式；自定义序数方案还需在条目中显式指定 `level_map`
和 `value_range`。默认是 `{"Low":0,"Medium":1,"High":2}` 和 `[0,2]`；
二元数值 0/1 使用 `"level_map": {}, "value_range": [0,1]`。
不同方法的同名数据集必须使用相同活动集合，避免不公平比较。

## 存储与验证

根据用户 2026-10-23 最新要求，本组指标代码、结果、论文图片、分析缓存及代码测试临时文件全部放在系统盘。
所有入口均检查输出真实路径和文件系统，只允许本目录的 `results/` 下实际位于系统盘的输出。
此前数据盘 `/data/wanzhen/experiments/prototype_discriminability/` 下的 99 个文件已迁至 `results/`，
逐文件 SHA256 校验后移除了原目录；迁移报告为 `results/results_migration_to_system_disk.json`。
历史结果元数据中的旧路径保留用于追溯；实际位置使用迁移报告中的目标目录。
默认原型输入仍从数据盘原项目读取。模型训练、模型测试产物放数据盘的长期约定不受本次指标分析调整影响。
入口把 Matplotlib 缓存放在系统盘 `results/`，禁用依赖代码的字节码缓存。
运行时仍建议使用 `python -B`。

综合入口依赖 numpy、pandas、matplotlib；测试另需 pytest，见 `requirements.txt`。
当前 zara 环境已安装，无需重新安装。

```bash
cd /home/wanzhen/attribute_prototype_analysis
PYTHONDONTWRITEBYTECODE=1 \
  /home/wanzhen/miniconda3/envs/zara/bin/python -B -m pytest tests \
  --basetemp=/home/wanzhen/attribute_prototype_analysis/results/test_tmp \
  -o cache_dir=/home/wanzhen/attribute_prototype_analysis/results/pytest_cache
```

指标仅描述语义原型的区分性，不能单独证明传感器上的属性可预测性或识别精度。
