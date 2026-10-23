# PAMAP2 最新属性细化与完整 A2 向量生成

入口：`scripts/run_pamap2_refinement_xxapi.py`。本说明中的命令需要由使用者手动执行。

默认参数：

- 接口：`https://xxapi.info/v1/chat/completions`，OpenAI 兼容格式。
- 模型：`gpt-6.1-sol`，`temperature=0.7`，非流式；`--stream` 可显式开启流式。
- 密钥：仅使用环境变量 `NEW_API_KEY`，不放入参数、配置或日志。
- 数据集与划分：PAMAP2，`split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json` 的 20 个 split。
- A1：`configs/attributes/activity_attribute_vectors_initial_pamap2_v1_5.json`，九属性、三级值。
- 最终属性数：9–15；不强制新增属性，以允许可预测性不确定时保留 A1。
- 请求超时 600 秒，每阶段最多 3 次尝试；429/524 重试至少等待 120 秒。

每个 split 依次执行：

1. 用最新 `planner1.py` 进行概念细化，考虑语义区分性、传感器可预测性、支持特征组及预测歧义。
   此阶段只输出语义计划，不生成活动级属性向量。
2. 用最新 `completion.py` 在一次响应中生成所有活动、所有最终 A2 属性的完整矩阵，
   包括重新赋值 `keep` 属性，绝不从 A1 补齐缺失列。
3. 所选 split 完成后，用 `normalized_ordinal` 距离重算 A1/A2 的 D_kNN、U_NN、U_Conf、JointGap。

完整首跑通常需要 40 次成功的 LLM 请求，失败重试可能增加请求数。脚本不训练分类器。

## 运行

```bash
cd /home/wanzhen/myproject_new
export NEW_API_KEY='在这里填写你的 xxapi 密钥'

/home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_pamap2_refinement_xxapi.py
```

先检查输入和参数，不写结果、不请求 API：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python \
  scripts/run_pamap2_refinement_xxapi.py --dry-run
```

先仅处理 split 13：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_pamap2_refinement_xxapi.py --split-indices 13
```

之后不带 `--split-indices` 运行，即可在相同配置下补齐其余 split。也可用
`--stage refine`、`--stage vectors`、`--stage compare` 分阶段执行；`vectors` 要求
相应 split 已有通过校验的计划，`compare` 要求已有通过校验的计划和完整向量。
调整温度示例为 `--temperature 0`；修改模型、温度、输入、实现或属性数范围时须
指定新的 `--root`，以保证实验结果可追溯。

## 输出与续跑

默认输出目录（`attrIbutes_v4` 中的 `I` 为大写）：

```text
configs/attrIbutes_v4/pamap2/
  experiment_config.json
  batch_status.json
  split_<原始编号>/
    a2_attribute_plan.json
    a2_attribute_plan.prompt.json
    a2_attribute_plan.prompt.txt
    a2_attribute_plan.refinement_evidence.json
    a2_attribute_plan.raw_response.txt
    activity_attribute_vectors_a2.json
    activity_attribute_vectors_a2.prompt.txt
    activity_attribute_vectors_a2.raw_response.txt
    activity_attribute_vectors_a2.raw_response.txt.generation.json
    activity_attribute_vectors_a2.report.json
    batch_logs/
    attempt_history/
  gap/
    a1_gap_normalized_ordinal_seed42.csv
    a1_gap_normalized_ordinal_seed42.meta.json
    selected_seed42_splits.json
    comparison/
      a1_gap.csv
      a1_gap.meta.json
      a2_gap.csv
      comparison.csv
      summary.json
```

重新运行相同命令会跳过通过 schema 与来源哈希校验的阶段，并尝试恢复已完成的
原始响应。旧增量向量不能成为新策略检查点。重试前备份已有阶段文件，单个 split
达到重试上限后记录失败并继续其余 split；`--fail-fast` 可改为立即停止。
同一输出目录有进程锁，避免两个批次同时写入。

两个数据集共用 `scripts/refinement_xxapi_common.py`，MHEALTH 入口为
`scripts/run_mhealth_refinement_xxapi.py`，输出到 `configs/attrIbutes_v4/mhealth/`。
初始 A1 输入仍来自 `configs/attributes/`，完整新版 A2 向量按 split 写入本目录。
