# 初始9维属性：MHEALTH motion_regularity 单属性语义重标注

本目录五份活动原型沿用 configs/attributes 的 initial v1.5。只有 MHEALTH 四个活动的 motion_regularity 从 High 改为 Medium；其余八维、其他活动/数据集及 initial_attributes.json 不变。

motion_regularity_revision.json 记录每个活动的判断、原始输入哈希、操作性标注规则与证据边界。Medium 是语义先验，不是实测时间规律性。未调用LLM、未读取IMU、未使用分类测试分数优化标签，也未改动运行配置或已保存模型。

覆盖率结果：/home/wanzhen/attribute_prototype_analysis/results/attribute_coverage_attributes_new
