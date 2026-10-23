# v4 属性向量：PAMAP2 / MHEALTH AutoGluon 默认集成 ZSL

USC-HAD、WISDM 的新增入口和完整流水线命令见 [USC-HAD / WISDM 使用说明](USCHAD_WISDM_V4_PIPELINE.md)。
共享实现支持这四个数据集；WISDM 使用 6 个未见类别，每个 split 为 180 个测试样本。

入口：

- `scripts/run_pamap2_v4_autogluon_default_ensemble.py`，默认物理 GPU 0。
- `scripts/run_mhealth_v4_autogluon_default_ensemble.py`，默认物理 GPU 1。

两个入口共用 `scripts/attributes_v4_autogluon_common.py`，均处理完整的 20 个 seed=42 split。
输入分别为 `configs/attrIbutes_v4/pamap2/`、`configs/attrIbutes_v4/mhealth/`，
逐 split 读取 `split_<原始编号>/activity_attribute_vectors_a2.json`，并检查完整生成的来源哈希。
属性集允许随 split 变化，属性值必须为 Low/Medium/High。

## 测试配置

| 项目 | PAMAP2 | MHEALTH |
|---|---|---|
| 活动划分 JSON | `split/zero_shot_split/pamap2_unseen4_random20_splits_seed42.json` | `split/zero_shot_split/mhealth_unseen4_random20_splits_seed42.json` |
| 测试样本 JSON | `split/zero_shot_split/pamap2_unseen4_random20_eval50_samples_seed42.json` | `split/zero_shot_split/mhealth_unseen4_random20_eval30_samples_seed42.json` |
| 每个 split 测试样本数 | 4 类 × 50 = 200 | 4 类 × 30 = 120 |

保持此前独立属性分类器实验的设置：`backend=autogluon`、`model_suite=default`、
`presets=medium_quality`、`fit_weighted_ensemble=true`、`num_bag_folds=0`、
`num_stack_levels=0`，每属性训练时限 300 秒、8 个 CPU、1 个可见 GPU。
活动识别仍用 `prototype_metric=ordinal_probability` 和 `attribute_weighting=validation_accuracy`。
分类器只从 seen 活动训练；属性准确率基于指定 JSON 中的 unseen 测试样本计算。

## 完整 nohup 命令

以下命令需手动执行，不需要 LLM API 密钥。

```bash
cd /home/wanzhen/myproject_new
mkdir -p /home/wanzhen/myproject_new/runtime/logs/v4_autogluon_default

nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/myproject_new/scripts/run_pamap2_v4_autogluon_default_ensemble.py train-test \
  --gpu-id 0 --time-limit 300 --num-cpus 8 --resume \
  --attribute-vectors-root /home/wanzhen/myproject_new/configs/attrIbutes_v4/pamap2 \
  --wait-for-vectors --poll-seconds 60 \
  > /home/wanzhen/myproject_new/runtime/logs/v4_autogluon_default/pamap2_gpu0.log 2>&1 < /dev/null &
echo $! > /home/wanzhen/myproject_new/runtime/logs/v4_autogluon_default/pamap2_gpu0.pid

nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/myproject_new/scripts/run_mhealth_v4_autogluon_default_ensemble.py train-test \
  --gpu-id 1 --time-limit 300 --num-cpus 8 --resume \
  --attribute-vectors-root /home/wanzhen/myproject_new/configs/attrIbutes_v4/mhealth \
  --wait-for-vectors --poll-seconds 60 \
  > /home/wanzhen/myproject_new/runtime/logs/v4_autogluon_default/mhealth_gpu1.log 2>&1 < /dev/null &
echo $! > /home/wanzhen/myproject_new/runtime/logs/v4_autogluon_default/mhealth_gpu1.pid
```

脚本在导入后端之前设置 `CUDA_VISIBLE_DEVICES`；每个进程中的逻辑 GPU 0 对应命令
指定的物理 GPU。`--wait-for-vectors` 会等待全部 20 组完整向量及校验报告可用，期间
不训练部分 split，也不调用 LLM。默认无限等待，可用 `--wait-timeout 21600` 限制为 6 小时。

日志查看：

```bash
tail -f /home/wanzhen/myproject_new/runtime/logs/v4_autogluon_default/pamap2_gpu0.log
tail -f /home/wanzhen/myproject_new/runtime/logs/v4_autogluon_default/mhealth_gpu1.log
```

## 统计与结果文件

默认输出目录：

```text
runtime/zshar_runs/pamap2/pamap2_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval50_seed42/
runtime/zshar_runs/mhealth/mhealth_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42/
```

保留原有 `summary.csv`、`aggregate_metrics.json` 及各 `split_<三位编号>/metrics.json`，
额外导出：

| 文件 | 含义 |
|---|---|
| `attribute_accuracy_by_split.csv` | 每个 split、每个属性的测试准确率及样本数 |
| `split_attribute_accuracy.csv` | 各 split 内所有属性准确率的等权平均 |
| `per_attribute_accuracy_summary.csv` | 每个属性跨 split 的均值、样本标准差、最小/最大值、覆盖数、缺失编号 |
| `attribute_accuracy_summary.json` | 整体均值、完成状态、未完成 split、逐属性汇总 |
| `v4_input_manifest.json` | 每组向量、定义及实验输入的内容哈希，防止标签变化后误用旧模型 |

整体属性平均准确率：先在每个 split 内对属性 accuracy 等权平均，再对 20 个 split
等权平均。逐属性均值只平均该属性实际存在且完成测试的 split；未出现的属性不填 0，
`present_split_count` 和 `coverage` 标明覆盖情况，未测试的 split 单独列出。
同名属性的定义变化由 `semantic_definition_variants` 标记。准确率为 [0,1] 小数。

训练成功返回前要求 20 个 split 的测试指标齐全。失败或中断时尽可能导出已完成部分，
并标记 `complete=false`。也可以单独重新汇总已有结果，不训练模型：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python \
  /home/wanzhen/myproject_new/scripts/run_pamap2_v4_autogluon_default_ensemble.py summarize
/home/wanzhen/miniconda3/envs/zara/bin/python \
  /home/wanzhen/myproject_new/scripts/run_mhealth_v4_autogluon_default_ensemble.py summarize
```

`check` 模式只做向量与测试样本检查，不等待、不训练、不写结果。
`--resume` 复用已完成模型，将中断的未完成模型目录归档后重训；模型复用仍会重新测试。
变更向量、测试输入或训练参数时需指定新的 `--run-name`，不能混入已有实验结果。

## 等待 GPU 资源释放

训练入口可添加 `--wait-for-gpu --gpu-poll-seconds 60 --gpu-max-used-mib 1024 --gpu-idle-checks 2`。
每 60 秒按物理 GPU 的 UUID 检查一次：连续两次没有其他计算进程，且已用显存不超过
1024 MiB，才开始训练。查询失败时继续等待，不会把失败当成空闲。
每张 GPU 的 v4 队列锁会保持到训练结束；不会终止任何已有 GPU 进程。
`--wait-for-vectors` 仍独立检查全部 20 组向量，未齐全时先等待向量。

USC-HAD/WISDM 一键流水线的训练阶段已默认启用这一等待方式。
等待日志为本次 nohup 指定的日志；开始检查 GPU 后，实验目录还会保存 `gpu_wait_status.json`。
