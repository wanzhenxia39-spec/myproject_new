# 属性原型距离：选择、匹配口径与本次结果

## 当前实验真正使用的匹配距离

已核对原项目入口：

- `/data/wanzhen/projects/myproject_new/scripts/run_pamap2_initial9_autogluon_default_ensemble.py:176`
- `/data/wanzhen/projects/myproject_new/scripts/run_mhealth_uschad_wisdm_dsads_initial9_autogluon_default.py:250`

这两类入口覆盖五数据集，显式设置 `prototype_metric="ordinal_probability"`、
`attribute_weighting="validation_accuracy"`。
匹配公式位于 `src/zshar/semantics/prototypes.py` 的 `match_prototype`；
`src/zshar/evaluation/scoring.py` 从验证报告读取属性准确率作为权重。
这里核对的是上述九维 A1 实验入口，不能根据通用配置类的 cosine 默认值推断实际实验。

令 `z(Low)=0, z(Medium)=0.5, z(High)=1`，样本属性预测概率为 `q_k(l)`，
候选活动属性原型为 `p_jk`，属性权重为 `w_k`。实际匹配距离为：

`D(x,j) = sum_k w_k * sum_l q_k(l)*|z(l)-z(p_jk)| / sum_k w_k`。

分类器的候选活动得分为 `1-D(x,j)`，因此最大得分与最小距离排序一致。

## 确定性原型之间怎样使用它

热力图没有样本预测概率，只有每个活动的确定性 Low/Medium/High 原型。
将活动 i 的第 k 个属性视为 `q_ik(l)=1[l=p_ik]`，上述公式严格化简为：

`D_proto(i,j) = sum_k w_k*|z(p_ik)-z(p_jk)| / sum_k w_k`。

**所有属性等权时，它就是原先的归一化 ordinal L1，数值不会改变。**
所以可以复用最终匹配的距离定义，但不能仅通过改名称得到一套更有区分性的数值，
也不能将本图写成“使用了真实分类器概率或经验混淆结果”。

本次五数据集主图使用 `ordinal_probability` 的确定性原型特例、统一等权，
不引入任何 split、样本预测或分类器验证结果；图色条和元数据均注明该口径。
这保持了对属性设计本身的描述，便于跨方案比较。

## 不同距离适用于什么问题

| 度量 | 公式（z 在 0–1，w 归一化） | 本任务的用途与限制 |
| --- | --- | --- |
| ordinal_probability，确定性原型 | `sum_k w_k*abs(z_ik-z_jk)` | 与当前最终匹配的基础代价一致；等权时与旧图相同。建议作为主图。 |
| ordinal_l1 / manhattan | `sum_k w_k*abs(z_ik-z_jk)` | 保留三级有序信息，解释直接；与上一行的原型特例等价。 |
| euclidean（归一化 RMS） | `sqrt(sum_k w_k*(z_ik-z_jk)^2)` | 相较小等级差，更强调少量大的等级差。已导出作为对照；不能在未验证识别效果时声称更优。 |
| validation_accuracy 加权 L1 | 第一行，w 来自指定验证报告 | 用于分析固定已训练模型的匹配空间；依赖模型、训练条件和 split。需提供对应实验的报告，不能与属性方案本身的无模型分析混淆。 |
| cosine | `1-dot(sqrt(w)*z_i,sqrt(w)*z_j)/(norm_i*norm_j)` | 主要比较方向，会丢失绝对等级强弱；不建议作为这套序数属性的主要区分性指标。 |
| hamming | `sum_k w_k*1[z_ik!=z_jk]` | 只数有几个属性不同；Low/Medium 与 Low/High 代价相同，可辅助解释属性覆盖但丢失序数间隔。 |

例如两个活动在所有属性上分别均为 Medium 与 High，它们的 cosine 距离为 0，
但它们并不是同一个属性原型。这也是这里不把 cosine 设为主图的原因。
没有具体概率证据时，不应将硬原型直接当作分布并盲目套用其他分布距离。

定义参考：[SciPy cityblock](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html)、
[SciPy euclidean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html)、
[SciPy cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)。
本脚本对 L1 除以权重和、对欧氏距离使用 RMS 归一化，以与项目匹配器一致并固定在 `[0,1]`。

## 权重文件的口径

`--weights-json` 可读取完整的属性权重字典，或现有验证报告中的：

```json
{"per_attribute": {"attribute_name": {"accuracy": 0.8}}}
```

上例仅演示结构；实际文件必须包含当前原型中的全部属性。
验证准确率必须在 `[0,1]`；零值与原匹配器一致，使用 `1e-10` 下限。
脚本不会默认用 1 补齐漏掉的属性，避免生成看似匹配实际实验、其实缺失权重的图。
归一化权重、原始权重、文件路径及哈希会记录在元数据中。
原型图仍不包含模型对真实样本的预测不确定性。

余弦的零向量本身无角度定义；本脚本为热力图显式约定两个零向量距离为 0，
一个为零、一个非零时为 1。该约定与项目匹配器的 EPS 实现并不在零向量情况完全等同，
因此不能把零向量余弦结果声称为原匹配器的严格复现。主图不使用 cosine。

## 本次导出

输入：五数据集最新版九维初始 A1 v1.5，所有活动，无 split。
根目录：`/home/wanzhen/attribute_prototype_analysis/results/distance_heatmaps/`。

- `<dataset>/`：主图，`ordinal_probability` 确定性原型特例，等权。
- `comparisons/euclidean/<dataset>/`：等权归一化欧氏 RMS 对照图。
- 每个目录均有 PDF、SVG、600 dpi PNG、原型矩阵、距离矩阵、逐类别对距离和元数据。
- `distance_summary.csv`：五数据集、两种度量的汇总；对角线不参与平均值。
- `batch_run_metadata.json`：本次输入和度量口径。
- `_archive/`：改动前的现有图、结果、旧代码及哈希清单。

所有图都用相同的 Blues 色阶 `[0,1]`，不会对某数据集单独拉伸数值范围。
