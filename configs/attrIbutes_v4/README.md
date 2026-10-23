# 最新版完整 A2 属性向量（v4）

路径大小写为 `configs/attrIbutes_v4/`。每个数据集单独一个目录：

- PAMAP2：`pamap2/split_<原始split编号>/activity_attribute_vectors_a2.json`
- MHEALTH：`mhealth/split_<原始split编号>/activity_attribute_vectors_a2.json`
- USC-HAD：`uschad/split_<原始split编号>/activity_attribute_vectors_a2.json`
- WISDM：`wisdm/split_<原始split编号>/activity_attribute_vectors_a2.json`
- DSADS：`dsads/split_<原始split编号>/activity_attribute_vectors_a2.json`

每个文件包含当前 split 下全部活动、全部最终属性的 Low/Medium/High 取值。
细化方案、提示词、原始响应、校验报告、日志及 GAP 指标保存在同一数据集目录下。
运行生成脚本后才会产生这些结果；空目录不代表向量已经生成。

```bash
cd /home/wanzhen/myproject_new
export NEW_API_KEY='你的xxapi密钥'
/home/wanzhen/miniconda3/envs/zara/bin/python -u scripts/run_pamap2_refinement_xxapi.py
/home/wanzhen/miniconda3/envs/zara/bin/python -u scripts/run_mhealth_refinement_xxapi.py
```

每个入口默认处理各自 seed=42 的 20 个 ZSL split：概念细化、一次性完整 A2 赋值、
归一化序数距离指标对比。支持 `--split-indices`、`--stage`、`--dry-run` 及断点续跑。
具体参数见 `scripts/PAMAP2_XXAPI_REFINEMENT.md`、`scripts/MHEALTH_XXAPI_REFINEMENT.md`。

USC-HAD、WISDM 的一键生成和训练入口为 `scripts/run_uschad_v4_pipeline.sh`、
`scripts/run_wisdm_v4_pipeline.sh`，分别默认使用 GPU 0、GPU 1。
完整 nohup 指令见 `scripts/USCHAD_WISDM_V4_PIPELINE.md`。WISDM 和 DSADS 使用 6 个未见活动，
每个 split 为 180 个测试样本；PAMAP2、MHEALTH、USC-HAD 使用 4 个未见活动。

DSADS 的属性生成与分类器训练分开执行，入口为 `scripts/run_dsads_refinement_xxapi.py`
和 `scripts/run_dsads_v4_autogluon_default_ensemble.py`；具体命令见
`scripts/DSADS_V4_REFINEMENT_AND_ZSL.md`。
