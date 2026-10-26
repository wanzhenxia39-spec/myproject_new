# 独立 Attribute Discrimination Coverage 计算

正式文件：`/home/wanzhen/attribute_prototype_analysis/attribute_discrimination_coverage.py`。
单文件仅依赖 Python 标准库，不需要其他本地代码、numpy、pandas、模型或预测文件。
综合分析入口也调用此文件中的唯一覆盖率公式实现。

## 公式与适用范围

`R_k = count(i<j: p_ik != p_jk) / [C(C-1)/2]`。

程序通过 `不同类别对数 = 总类别对数 - sum_l n_l(n_l-1)/2` 精确计算。
不含自身配对，不重复统计，不按等级差加权，不做距离或 min-max 归一化。
覆盖率比率以 0–1 保存，百分比请乘以 100。

支持二元、三级、多级或其他离散标签，属性数量不限于九维。
默认只判断原标签是否相同；数值字符串与等值数字一致，例如 CSV 的 `1` 与 JSON 的 `1.0`。
Low/Medium/High 不会自动映射到数字；混合编码时可在清单条目中显式提供 `level_map`。
缺失值、NaN、重复活动/属性和不完整原型会报错，不静默填补。

每份输入至少有两个活动，每个活动必须包含相同的非空属性集合。
同一方法跨数据集必须使用相同属性名，矩阵按名称对齐，保留第一份输入的属性顺序。
不同方法可有完全不同的属性名和维度，分别生成自己的矩阵。
比较同一数据集的不同方法时，需要使用相同活动集合。

## 当前九维 A1 的默认运行

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/attribute_discrimination_coverage.py \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/attribute_coverage_a1_v1_5
```

默认读取五数据集的 `activity_attribute_vectors_initial_<dataset>_v1_5.json`，
分别计算每个属性的覆盖率，生成九行 × 五列的矩阵，不涉及 split。
该入口只计算和导出 CSV/JSON，不绘图、不计算活动原型距离或完整原型 APCR。

## 一个其他方法的原型文件

JSON 输入示例：

```json
{
  "activity A": {"attribute_1": "Low", "attribute_2": "High"},
  "activity B": {"attribute_1": "Medium", "attribute_2": "Low"}
}
```

CSV 输入首列必须为 `activity`，其余列是属性：

```csv
activity,attribute_1,attribute_2
activity A,Low,High
activity B,Medium,Low
```

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/attribute_discrimination_coverage.py \
  --vectors /实际路径/imuzero_pamap2.json --method imuzero --dataset pamap2 \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/imuzero_coverage
```

上例仅指定自备原型位置，不预设 IMUZero 具体属性或编码。

## 多数据集和多方法矩阵

准备清单 `methods.json`，每项的输入路径相对清单所在目录解析，也支持绝对路径：

```json
[
  {"method": "imuzero", "dataset": "pamap2", "vectors": "imuzero_pamap2.json"},
  {"method": "imuzero", "dataset": "mhealth", "vectors": "imuzero_mhealth.csv"},
  {"method": "other_method", "dataset": "pamap2", "vectors": "other_pamap2.json"},
  {"method": "other_method", "dataset": "mhealth", "vectors": "other_mhealth.json"}
]
```

添加其余三个数据集的条目即可形成五列矩阵。
方法和数据集标识使用字母、数字、下划线或连字符，不允许重复的方法/数据集组合。

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -B \
  /home/wanzhen/attribute_prototype_analysis/attribute_discrimination_coverage.py \
  --manifest /实际路径/methods.json \
  --output-dir /home/wanzhen/attribute_prototype_analysis/results/method_coverage_comparison
```

每个方法的输出为：

```text
<output>/
  all_attribute_metrics.csv
  dataset_summary.csv
  run_metadata.json
  <method>/
    attribute_coverage_matrix.csv   # K 行 × D 列，另有属性名索引列
    attribute_dataset_means.csv     # 每个属性跨数据集的等权平均覆盖率
    <dataset>/
      attribute_metrics.csv         # 不同/相同类别对数、总类别对数、覆盖率
      summary.json
```

本机 CLI 输出限定在系统盘 `/home/wanzhen/attribute_prototype_analysis/results/` 下。
如在其他机器使用，可修改文件顶部 `RESULTS_ROOT`；纯计算 API 与文件存储路径无关。

## Python API

```python
from attribute_discrimination_coverage import (
    read_vectors, compute_attribute_coverage, build_coverage_matrix,
)

# 一个数据集：返回每个属性的类别对计数和覆盖率。
vectors = read_vectors('/实际路径/imuzero_pamap2.json')
metrics = compute_attribute_coverage(vectors)

# 多个数据集：返回行名、列名和二维数值列表，无需第三方库。
matrix = build_coverage_matrix({
    'PAMAP2': vectors,
    'MHEALTH': read_vectors('/实际路径/imuzero_mhealth.json'),
})
print(matrix['attributes'])
print(matrix['datasets'])
print(matrix['values'])
```

该指标只判断每个属性能区分多少活动类别对，不评价传感器可预测性、
属性间互补性或分类准确率。属性等级数和活动类别组成不同，也会影响覆盖率及其上限。
