<!-- MHEALTH_INITIAL_REVISION_START -->
## MHEALTH 初始原型已更新

MHEALTH 初始9维输入现为 `configs/attributes_new/activity_attribute_vectors_initial_mhealth_v1_5.json`。
已覆盖 `before/mhealth/` 中的 L1 / RMS 热力图、距离矩阵和向量快照，并同步更新 MHEALTH 对照图及依赖初始基线的汇总。
其他数据集和 A2 原型、A2 距离均值/标准差保持原值。A2 的生成依据仍是旧版 A1，因此当前对照图展示“新版初始原型”与“既有 A2”的几何比较，不能描述为从新版初始向量重新细化得到的结果。
更新核验见 `before/mhealth/revision_verification.json`；重现当前覆盖操作：
`python -B /home/wanzhen/attribute_prototype_analysis/update_mhealth_initial_distance.py`。
下面原始批处理命令复现的是历史 A1/A2 分析，需再执行本更新脚本才能得到此处的新版 MHEALTH 初始结果。
<!-- MHEALTH_INITIAL_REVISION_END -->

# 最新属性向量：细化前后原型距离与属性区分覆盖率

本目录位于真实系统盘 `/home/wanzhen/attribute_prototype_analysis/new_results/`。
代码为上一级的 `compare_refinement_prototypes.py`；缓存、日志和测试临时文件也保存在本目录。

## 输入版本与范围

- 细化前：五个数据集的九维 `activity_attribute_vectors_initial_<dataset>_v1_5.json`。
- 细化后：当前 `configs/attrIbutes_v4/<dataset>/split_*/activity_attribute_vectors_a2.json`，
  每数据集全部 20 个 seed=42 split，共 100 份 A2。
- 数据集：PAMAP2、MHEALTH、USC-HAD、WISDM、DSADS；分别有 12、12、12、18、19 个活动。
- 没有使用 DSADS 的旧备份、PAMAP2 二级/五级消融向量或 MHEALTH 的旧版细化目录。
- 已核对细化配置中的 A1 SHA256 和 split 清单，并检查 A2 属性与对应细化计划一致。
- 计算对象是每份向量中的全部活动类别，不读取传感器窗口、模型预测或 GZSL 指标。

## 优先查看的图片

五数据集前后距离对比图（左 A1，右 A2 的 20 split 距离均值）：

| 数据集 | 归一化序数 L1 主图 | 欧氏 RMS 对照 |
| --- | --- | --- |
| PAMAP2 | [PNG](comparison/pamap2/prototype_distance_before_after.png) · [PDF](comparison/pamap2/prototype_distance_before_after.pdf) | [PNG](comparison/pamap2/euclidean_prototype_distance_before_after.png) |
| MHEALTH | [PNG](comparison/mhealth/prototype_distance_before_after.png) · [PDF](comparison/mhealth/prototype_distance_before_after.pdf) | [PNG](comparison/mhealth/euclidean_prototype_distance_before_after.png) |
| USC-HAD | [PNG](comparison/uschad/prototype_distance_before_after.png) · [PDF](comparison/uschad/prototype_distance_before_after.pdf) | [PNG](comparison/uschad/euclidean_prototype_distance_before_after.png) |
| WISDM | [PNG](comparison/wisdm/prototype_distance_before_after.png) · [PDF](comparison/wisdm/prototype_distance_before_after.pdf) | [PNG](comparison/wisdm/euclidean_prototype_distance_before_after.png) |
| DSADS | [PNG](comparison/dsads/prototype_distance_before_after.png) · [PDF](comparison/dsads/prototype_distance_before_after.pdf) | [PNG](comparison/dsads/euclidean_prototype_distance_before_after.png) |

覆盖率矩阵：

- [原九个属性的细化前后对比](comparison/shared_attributes_coverage_before_after.png)：9 属性 × 5 数据集，
  右图 `n` 是细化后实际包含该属性的 split 数量。
- [细化前完整矩阵](before/attribute_coverage_heatmap.png)。
- 细化后完整属性并集矩阵：`after/attribute_coverage_heatmap_page_*.png`，按 24 行分页；
  同时保存 PDF/SVG。每个有效单元格标注均值和参与平均的 split 数。
- 每数据集的完整“属性 × 20 split”矩阵：
  [PAMAP2](after/pamap2/attribute_coverage_by_split.png)、
  [MHEALTH](after/mhealth/attribute_coverage_by_split.png)、
  [USC-HAD](after/uschad/attribute_coverage_by_split.png)、
  [WISDM](after/wisdm/attribute_coverage_by_split.png)、
  [DSADS](after/dsads/attribute_coverage_by_split.png)。

所有图统一使用 Blues、固定 `[0,1]` 色阶、两位小数标注；PNG 为 300 dpi，
PDF/SVG 为矢量文件。灰色 `N/A` 表示没有该属性，白色 `0.00` 才表示实际覆盖率为零。

## 公式与汇总口径

令 Low/Medium/High 分别编码为 `z=0/0.5/1`，每份原型有 K 个属性、C 个活动。

- 距离主图：`D(i,j)=mean_k |z_ik-z_jk|`，即现有 `ordinal_probability`
  在确定性原型和等权条件下的特例，也就是归一化序数 L1。
- 欧氏对照：`D_RMS(i,j)=sqrt(mean_k (z_ik-z_jk)^2)`。
- 属性覆盖率：`R_k=count(i<j: p_ik != p_jk)/binom(C,2)`，不考虑等级差大小。
- 所有距离均不使用验证准确率加权；输入为纯语义原型。
- A2 每个 split 使用自己的真实属性维度计算距离，再对 20 份距离矩阵逐单元格取均值。
  不平均 Low/Medium/High 标签，也不把不同名称的属性强行拼接成统一原型。
- 覆盖率按**精确属性名**对齐；一个属性在某 split 没有出现时保持缺失，不补零。
  覆盖率均值/总体标准差仅在包含该属性的 split 上计算，并另存出现次数。
- 没有合并名称相似但不相同的属性；矩阵按名称对齐不代表额外证明了定义完全等价。
- `dataset_summary.csv` 中细化后平均覆盖率先在每个 split 的实际属性中平均，
  再对 20 个 split 等权平均；它与“全部属性并集的条件均值再平均”不同。
- 活动距离的均值只取不重复的非对角活动对；不同 K 均归一化到 `[0,1]`。

数值可以上升也可以下降。本分析描述语义原型的几何差异与单属性区分性，
不能仅据此推导传感器属性预测能力或 GZSL 识别精度。

## 数值与溯源文件

- `dataset_summary.csv`：五个数据集的前后平均距离、平均覆盖率及属性数范围。
- `all_attribute_coverage.csv`：全部 105 份输入逐属性的精确类别对计数与覆盖率。
- `before/attribute_coverage_matrix.csv`：A1 的九属性 × 五数据集。
- `after/attribute_coverage_matrix.csv`：A2 完整属性并集 × 五数据集的条件均值。
- `after/attribute_presence_counts.csv`、`after/attribute_coverage_std.csv`：出现次数与总体标准差。
- `after/<dataset>/distance_matrix.csv`、`distance_matrix_std.csv`：20 split 的 L1 距离均值和标准差。
- 同目录 `euclidean_*`：欧氏 RMS 数值；`*_distance_change_matrix.csv` 为细化后减细化前。
- `after/<dataset>/splits/split_<id>/`：每个 split 的输入向量快照、归一化原型、覆盖率和两种距离。
- `run_metadata.json`：输入路径、SHA256、属性清单、公式、汇总方式和代码哈希。
- `verification_report.json`：105 份原型表的独立类别对枚举、距离对称性、零对角、值域及磁盘检查。
- `analysis.log`、`test.log`：分析日志和聚合逻辑的测试结果。

## 重现命令

```bash
cd /home/wanzhen/attribute_prototype_analysis
PYTHONDONTWRITEBYTECODE=1 \
  /home/wanzhen/miniconda3/envs/zara/bin/python -B compare_refinement_prototypes.py \
  --output-dir /home/wanzhen/attribute_prototype_analysis/new_results/reproduce
```

已完成的输出目录不会被覆盖；再次运行使用 `new_results/` 下新的子目录。


## 已迁移的初始9维模型实验结果

2026-10-23：属性可预测性与ZSL逐属性移除两组既有结果已迁移至本系统盘目录。

- [两组结果入口与图片、表格索引](INITIAL9_RESULTS_INDEX.md)
- [属性可预测性 Excel](initial9_seen_predictability/attribute_predictability_9x5.xlsx)
- [ZSL逐属性移除 Excel](initial9_zsl_loao/initial9_zsl_loao_results.xlsx)
