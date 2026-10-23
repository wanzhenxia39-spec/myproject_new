# USC-HAD / WISDM v4：属性细化、完整 A2 向量和 AutoGluon ZSL

默认使用 xxapi 的 `gpt-6.1-sol`、`temperature=0.7`、非流式请求，只读取 `NEW_API_KEY`。
每个数据集处理 `split/zero_shot_split/` 下 seed=42 的全部 20 个 split。

| 项目 | USC-HAD | WISDM |
|---|---|---|
| 脚本数据集名称 | `uschad` | `wisdm` |
| 默认物理 GPU | 0 | 1 |
| 未见活动数 | 4 | 6 |
| 每类测试样本数 | 30 | 30 |
| 每个 split 测试样本总数 | 120 | 180 |
| 新版向量目录 | `configs/attrIbutes_v4/uschad/` | `configs/attrIbutes_v4/wisdm/` |

实际读取的测试配置：

- USC-HAD：`uschad_unseen4_random20_splits_seed42.json` 和 `uschad_unseen4_random20_eval30_samples_seed42.json`。
- WISDM：`wisdm_unseen6_random20_splits_seed42.json` 和 `wisdm_unseen6_random20_eval30_samples_seed42.json`。

## 完整 nohup 指令

以下命令由使用者手动执行，每个后台任务依次完成生成与训练：

```bash
cd /home/wanzhen/myproject_new
export NEW_API_KEY='你的xxapi密钥'
mkdir -p /home/wanzhen/myproject_new/runtime/logs/v4_full_pipeline

nohup bash /home/wanzhen/myproject_new/scripts/run_uschad_v4_pipeline.sh \
  --gpu-id 0 --time-limit 300 --num-cpus 8 \
  > /home/wanzhen/myproject_new/runtime/logs/v4_full_pipeline/uschad_gpu0.log 2>&1 < /dev/null &
echo $! > /home/wanzhen/myproject_new/runtime/logs/v4_full_pipeline/uschad_gpu0.pid

nohup bash /home/wanzhen/myproject_new/scripts/run_wisdm_v4_pipeline.sh \
  --gpu-id 1 --time-limit 300 --num-cpus 8 \
  > /home/wanzhen/myproject_new/runtime/logs/v4_full_pipeline/wisdm_gpu1.log 2>&1 < /dev/null &
echo $! > /home/wanzhen/myproject_new/runtime/logs/v4_full_pipeline/wisdm_gpu1.pid
```

两个任务的 LLM 请求可并行进行。每个数据集内部按 split 顺序执行：

1. 最新 `planner1.py` 生成细化方案，要求考虑可预测性、支持特征组、预期传感器证据和预测歧义。
2. `completion.py` 一次生成完整活动 × A2 属性矩阵，包括重新赋值 `keep` 属性，9–15 个三级属性。
3. 计算等权 `normalized_ordinal` 距离下的 A1/A2 D_kNN、U_NN、U_Conf、JointGap 对比。
4. 仅在生成阶段成功完成后启动 AutoGluon 默认集成独立属性分类器，训练并测试 20 个 split。
5. 导出活动识别结果、逐属性跨 split 的准确率和整体属性平均准确率。

每数据集通常需要 40 次成功的 LLM 请求，失败重试可能增加请求次数。超时、重试、
完整矩阵校验和断点恢复沿用 PAMAP2/MHEALTH 的实现。生成阶段失败时整条流水线返回
非零状态，不会继续训练。重跑相同命令会恢复已校验的生成结果和已完成模型。

### SSL EOF / 代理连接失败时补齐 WISDM

生成脚本默认 `--proxy-mode direct`，只为 API 主机增加 `NO_PROXY/no_proxy`，
绕过环境中的 HTTP/HTTPS/ALL 代理，仍校验 TLS 证书。需要使用环境代理时，
直接 Python 入口可指定 `--proxy-mode environment`。网络路由调整不改变模型、
提示词和已完成 split 的有效性。已运行进程需要退出后重新启动才能加载修改。

```bash
cd /home/wanzhen/myproject_new
# 在此终端已设置 NEW_API_KEY 后执行；不要同时运行两个同目录的生成任务。
mkdir -p runtime/logs/v4_full_pipeline
nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_wisdm_refinement_xxapi.py \
  --root /home/wanzhen/myproject_new/configs/attrIbutes_v4/wisdm \
  --proxy-mode direct --max-attempts 6 --retry-delay 60 \
  > runtime/logs/v4_full_pipeline/wisdm_refinement_retry_direct.log 2>&1 < /dev/null &
echo $! > runtime/logs/v4_full_pipeline/wisdm_refinement_retry_direct.pid
```

默认遍历全部 20 个 split，跳过通过校验的方案和向量，只补齐失败部分，并更新完整对比。
已有带 `--wait-for-vectors --wait-for-gpu` 的训练任务无需重启：补齐后继续等待 GPU 空闲。

## 分阶段入口

一键脚本支持 `--stage generate`（只生成、重算 GAP）、`--stage train`（只训练测试）。
其余参数包括 `--attribute-vectors-root`、`--model`、`--base-url`、`--temperature`、
`--time-limit`、`--num-cpus`；默认 Python 为 `/home/wanzhen/miniconda3/envs/zara/bin/python`，
可用环境变量 `V4_PYTHON_BIN` 覆盖。

直接的 Python 入口：

```text
scripts/run_uschad_refinement_xxapi.py
scripts/run_wisdm_refinement_xxapi.py
scripts/run_uschad_v4_autogluon_default_ensemble.py
scripts/run_wisdm_v4_autogluon_default_ensemble.py
```

生成入口可使用 `--dry-run`，训练入口可使用 `check` 只读检查全部向量和测试样本。
训练入口 `summarize` 可单独更新已有模型测试结果的属性统计。

## 输出

每个 split 的完整向量：

```text
configs/attrIbutes_v4/uschad/split_<原始编号>/activity_attribute_vectors_a2.json
configs/attrIbutes_v4/wisdm/split_<原始编号>/activity_attribute_vectors_a2.json
```

默认训练设置为 `model_suite=default`、`medium_quality`、加权集成启用、bagging/stacking
层数为 0，每属性时限 300 秒。活动原型匹配为 `ordinal_probability`、
`attribute_weighting=validation_accuracy`，与此前 PAMAP2/MHEALTH 相同。

训练测试结果目录：

```text
runtime/zshar_runs/uschad/uschad_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42/
runtime/zshar_runs/wisdm/wisdm_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42/
```

包含 `summary.csv`、`aggregate_metrics.json` 和逐 split 测试详情，并额外保存：

- `per_attribute_accuracy_summary.csv`：每属性跨 split 均值、标准差、覆盖数和缺失编号。
- `attribute_accuracy_by_split.csv`：逐 split、逐属性的测试准确率。
- `split_attribute_accuracy.csv`：逐 split 内全部属性的平均准确率。
- `attribute_accuracy_summary.json`：整体平均准确率及 20 个 split 的完成状态。

统计口径与此前一致：整体先在 split 内对属性求均值，再对 20 个 split 等权平均；
逐属性均值只统计实际出现的 split，缺失属性不记 0。同名属性存在定义差异时，
`semantic_definition_variants` 会标明定义版本数量。

日志查看：

```bash
tail -f /home/wanzhen/myproject_new/runtime/logs/v4_full_pipeline/uschad_gpu0.log
tail -f /home/wanzhen/myproject_new/runtime/logs/v4_full_pipeline/wisdm_gpu1.log
```
