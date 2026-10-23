# WISDM v4：seed=43 属性细化和 AutoGluon ZSL

当前 v4 向量是按 seed=42 的类别划分分别细化的。seed=43 的 20 个 split 与其没有
相同的划分编号，所以先为 seed=43 独立生成 A2，再训练测试。

两个入口固定 seed=43，默认分别读取以下配置，不会重新抽样测试窗口：

- `split/zero_shot_split/wisdm_unseen6_random20_splits_seed43.json`
- `split/zero_shot_split/wisdm_unseen6_random20_eval30_samples_seed43.json`

## 1. 生成细化方案及完整 A2 向量

沿用最新可预测性提示词和约束，xxapi `/v1/chat/completions`、`gpt-6.1-sol`、
`temperature=0.7`、非流式、`NEW_API_KEY`；默认直连且验证 TLS。
每个 split 先生成语义方案，再用一次响应生成全部活动、全部最终属性的三级向量，
keep 属性也重新赋值。属性数 9–15；GAP 比较使用等权 normalized_ordinal 距离。

```bash
cd /home/wanzhen/myproject_new
export NEW_API_KEY='你的xxapi密钥'
mkdir -p runtime/logs/v4_wisdm_seed43

nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_wisdm_refinement_xxapi_seed43.py \
  --proxy-mode direct --max-attempts 6 --retry-delay 60 \
  > runtime/logs/v4_wisdm_seed43/refinement.log 2>&1 < /dev/null &
echo $! > runtime/logs/v4_wisdm_seed43/refinement.pid
```

每个 split 输出位于
`configs/attrIbutes_v4/wisdm/seed43/split_<编号>/activity_attribute_vectors_a2.json`。
保留细化方案、原始响应及校验报告；同命令重跑会跳过通过校验的阶段。
模型请求只在实际运行时发起。`--dry-run` 可先检查输入和配置，不写文件也不调用 API。

监控：`tail -f runtime/logs/v4_wisdm_seed43/refinement.log`。

## 2. 在 GPU 1 训练并测试

生成日志显示 `Completed stage=all, 20 splits` 后，先检查：

```bash
cd /home/wanzhen/myproject_new
/home/wanzhen/miniconda3/envs/zara/bin/python \
  scripts/run_wisdm_v4_autogluon_default_ensemble_seed43.py check --gpu-id 1
```

检查通过后启动：

```bash
cd /home/wanzhen/myproject_new
mkdir -p runtime/logs/v4_wisdm_seed43
nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_wisdm_v4_autogluon_default_ensemble_seed43.py train-test \
  --gpu-id 1 --time-limit 300 --num-cpus 8 --resume \
  > runtime/logs/v4_wisdm_seed43/autogluon_gpu1.log 2>&1 < /dev/null &
echo $! > runtime/logs/v4_wisdm_seed43/autogluon_gpu1.pid
```

默认直接使用 GPU 1，不等待 GPU 完全空闲。若要在生成未完成时先挂起训练进程，
在训练命令中增加 `--wait-for-vectors --poll-seconds 60`；它只等待向量齐全。
自动校验计划中的 seen/unseen 类别、矩阵、来源哈希及固定测试窗口。

每 split 为 6 个未见类别 × 30 样本，共 180 个；只用 seen 类训练。分类器使用
AutoGluon 默认模型集、medium_quality、加权集成、bagging=0、stacking=0；
每属性 300 秒、8 CPU。原型匹配为 ordinal_probability，权重为 validation_accuracy。

监控：`tail -f runtime/logs/v4_wisdm_seed43/autogluon_gpu1.log`。

结果目录：

```text
runtime/zshar_runs/wisdm/wisdm_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed43/
```

保存活动识别 `aggregate_metrics.json`、`summary.csv`，并导出：

- `attribute_accuracy_summary.json`：先 split 内平均属性准确率，再对 20 个 split 等权平均。
- `split_attribute_accuracy.csv`：每 split 的平均属性准确率。
- `per_attribute_accuracy_summary.csv`：各属性跨 split 的均值、标准差及覆盖数；缺失不计 0。
- `attribute_accuracy_by_split.csv`：逐属性逐 split 的测试准确率。

只汇总已有结果可运行训练入口的 `summarize` 模式，不会重新训练。
