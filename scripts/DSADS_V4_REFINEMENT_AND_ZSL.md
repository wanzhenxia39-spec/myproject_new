# DSADS v4：分开进行属性细化和 AutoGluon ZSL

迁移到数据盘后的最新版 A2 独立重测指令见 [DSADS_V4_RERUN_DATA.md](DSADS_V4_RERUN_DATA.md)。
该指令显式使用 `/data/wanzhen/projects/myproject_new`，并为重测设置独立结果目录。

两个独立入口：

- `scripts/run_dsads_refinement_xxapi.py`：细化方案、完整 A2 向量、距离指标对比。
- `scripts/run_dsads_v4_autogluon_default_ensemble.py`：基于这些向量训练和测试属性分类器。

使用 `split/zero_shot_split/` 下固定的 seed=42 配置：

```text
dsads_unseen6_random20_splits_seed42.json
dsads_unseen6_random20_eval30_samples_seed42.json
```

完整运行 20 个 split。每个 split 有 6 个未见类别，每类 30 个固定测试窗口，共 180 个
测试样本；测试窗口已与实际标签进行只读校验。不会重新随机抽样。

## 第一步：只进行属性细化与完整 A2 向量生成

默认沿用 xxapi、`gpt-6.1-sol`、`temperature=0.7`、非流式请求，密钥来自 `NEW_API_KEY`。
细化考虑语义区分性、IMU/特征可预测性、支持特征组及预测歧义，保留现有 A1 定义和
操作谱系约束，输出 9–15 个三级属性。不确定新增价值时允许保持全部 A1 概念。

```bash
cd /home/wanzhen/myproject_new
export NEW_API_KEY='你的xxapi密钥'
mkdir -p /home/wanzhen/myproject_new/runtime/logs/v4_dsads

nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/myproject_new/scripts/run_dsads_refinement_xxapi.py \
  --root /home/wanzhen/myproject_new/configs/attrIbutes_v4/dsads \
  --model gpt-6.1-sol --temperature 0.7 \
  > /home/wanzhen/myproject_new/runtime/logs/v4_dsads/refinement.log 2>&1 < /dev/null &
echo $! > /home/wanzhen/myproject_new/runtime/logs/v4_dsads/refinement.pid
```

每个 split 先输出语义计划，再通过一次响应生成全部活动 × 全部最终属性的完整矩阵。
`keep` 属性的值也重新赋值，不从 A1 复制或补齐。最后重算 `normalized_ordinal` 距离下
的 D_kNN、U_NN、U_Conf、JointGap；这一入口不会自动启动分类器训练。

向量输出路径为：

```text
configs/attrIbutes_v4/dsads/split_<原始编号>/activity_attribute_vectors_a2.json
```

细化方案、提示词、原始响应、校验报告和 `batch_status.json` 位于同一数据集目录。
失败请求会重试，重新运行同一命令可续跑。原始编号不是 1–20 连续编号。

监控：

```bash
tail -F /home/wanzhen/myproject_new/runtime/logs/v4_dsads/refinement.log
```

等待日志出现 `Completed stage=all, 20 splits` 后，再手动执行第二步。
也可先进行只读检查：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python \
  /home/wanzhen/myproject_new/scripts/run_dsads_v4_autogluon_default_ensemble.py check --gpu-id 0
```

## 第二步：单独启动 AutoGluon 属性分类器训练与 ZSL 测试

此步骤不需要 API 密钥。默认物理 GPU 0，可把 `--gpu-id 0` 改为 `--gpu-id 1`。

也可使用独立启动入口 `scripts/run_dsads_v4_zsl.sh`，默认训练测试、GPU 0、
每属性 300 秒、8 个 CPU，并启用 `--resume`。它会读取当前完整 A2 向量，
不等待 GPU 空闲；其他参数转交给 Python 入口，命令行指定值覆盖默认值。
已完成模型仅在属性向量和测试输入的哈希均一致时允许复用；A1 实验使用不同结果目录。

```bash
cd /home/wanzhen/myproject_new
bash scripts/run_dsads_v4_zsl.sh check
mkdir -p runtime/logs/v4_dsads
nohup bash scripts/run_dsads_v4_zsl.sh --gpu-id 0 \
  > runtime/logs/v4_dsads/autogluon_v4_gpu0.log 2>&1 < /dev/null &
echo $! > runtime/logs/v4_dsads/autogluon_v4_gpu0.pid
```

监控：`tail -f runtime/logs/v4_dsads/autogluon_v4_gpu0.log`。
如需保留旧 A2 实验并从头训练，可给新任务增加唯一的 `--run-name`。

### 等待 GPU 0 上的 USC-HAD 完成后自动运行

给 shell 入口增加 `--after-uschad`。它每 60 秒检查 USC-HAD 的进程及固定 seed-42
实验目录，只有进程退出、20 个 split 的测试结果齐全且属性准确率汇总完整后才继续。
如果 USC-HAD 异常退出且结果不完整，队列任务报错退出，不会直接启动 DSADS。
它不等待 GPU 上其他任务结束，也不依赖 WISDM 的完成状态。

```bash
cd /home/wanzhen/myproject_new
mkdir -p runtime/logs/v4_dsads
nohup bash scripts/run_dsads_v4_zsl.sh --gpu-id 0 --after-uschad \
  > runtime/logs/v4_dsads/autogluon_v4_gpu0_after_uschad.log 2>&1 < /dev/null &
echo $! > runtime/logs/v4_dsads/autogluon_v4_gpu0_after_uschad.pid
```

日志先记录 USC-HAD 的完成进度，通过检查后由同一后台任务开始 DSADS 训练测试。

等价的完整 Python 调用如下（与上面的 shell 入口二选一执行）：

```bash
cd /home/wanzhen/myproject_new
mkdir -p /home/wanzhen/myproject_new/runtime/logs/v4_dsads

nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /home/wanzhen/myproject_new/scripts/run_dsads_v4_autogluon_default_ensemble.py train-test \
  --gpu-id 0 --time-limit 300 --num-cpus 8 --resume \
  --attribute-vectors-root /home/wanzhen/myproject_new/configs/attrIbutes_v4/dsads \
  > /home/wanzhen/myproject_new/runtime/logs/v4_dsads/autogluon_gpu0.log 2>&1 < /dev/null &
echo $! > /home/wanzhen/myproject_new/runtime/logs/v4_dsads/autogluon_gpu0.pid
```

默认不等待缺失向量：全部 20 组矩阵、计划和来源校验均通过后才开始训练。
如果想主动等待另一进程完成生成，可以添加 `--wait-for-vectors --poll-seconds 60`。

训练配置与此前数据集相同：AutoGluon 默认模型集、`medium_quality`、启用加权集成，
bagging/stacking 层数为 0，每属性时限 300 秒、8 个 CPU。只使用 seen 类训练。
原型匹配采用 `ordinal_probability`、`attribute_weighting=validation_accuracy`。

监控：

```bash
tail -F /home/wanzhen/myproject_new/runtime/logs/v4_dsads/autogluon_gpu0.log
```

## 结果及属性准确率

默认结果目录：

```text
runtime/zshar_runs/dsads/dsads_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42/
```

包括活动识别的 `summary.csv`、`aggregate_metrics.json` 和各 split 的测试详情；另有：

- `attribute_accuracy_summary.json`：20 个 split 的整体平均属性准确率与完成状态。
- `per_attribute_accuracy_summary.csv`：各属性跨 split 平均准确率、标准差和覆盖数。
- `split_attribute_accuracy.csv`：每个 split 内全部属性的平均准确率。
- `attribute_accuracy_by_split.csv`：逐 split、逐属性的准确率。

整体采用先在 split 内平均，再对 20 个 split 等权平均的口径。逐属性均值仅统计该
属性实际出现的 split，未出现不计 0。成功结束要求 20 个 split 的测试结果齐全；
中断时尽可能保存带 `complete=false` 的部分统计，`--resume` 可恢复模型并重新测试。

仅更新已有结果的统计，不训练：

```bash
/home/wanzhen/miniconda3/envs/zara/bin/python \
  /home/wanzhen/myproject_new/scripts/run_dsads_v4_autogluon_default_ensemble.py summarize
```
