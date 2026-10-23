# DSADS 最新 A2：迁移到数据盘后的独立重测

入口为 `/data/wanzhen/projects/myproject_new/scripts/run_dsads_v4_zsl.sh`。
该入口显式传入数据盘结果目录，并将临时文件、Joblib 和 Torch 缓存放到项目的
`runtime/tmp` 和 `runtime/cache` 下。通过旧 `/home` 符号链接调用时也解析到物理目录。

输入固定为：

- `configs/attrIbutes_v4/dsads/split_<编号>/activity_attribute_vectors_a2.json`。
- `split/zero_shot_split/dsads_unseen6_random20_splits_seed42.json`。
- `split/zero_shot_split/dsads_unseen6_random20_eval30_samples_seed42.json`。

本轮独立结果名使用
`dsads_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42_refined_rerun_20261021`，
避免复用旧版 A2 的结果。中断后保持此名称和 `--resume` 即可恢复；若再次重生成向量，
应另设新的 `--run-name`，已有模型不会在输入哈希变化后继续使用。

## 只读校验

```bash
bash /data/wanzhen/projects/myproject_new/scripts/run_dsads_v4_zsl.sh check \
  --gpu-id 1 \
  --run-name dsads_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42_refined_rerun_20261021
```

校验全部 20 个 split 的 A2 来源、属性计划、完整矩阵、固定测试窗口及已有结果兼容性。
此命令不训练或调用 LLM。

## 后台训练与测试

以下使用 GPU 1；也可将 `--gpu-id 1` 改成 `--gpu-id 0`。默认立即启动，不等待 GPU 空闲。
如需排队，在 `--gpu-id 1` 后增加 `--wait-for-gpu --gpu-poll-seconds 60`。
不要对同一结果目录同时启动多个任务。

```bash
cd /data/wanzhen/projects/myproject_new
mkdir -p /data/wanzhen/projects/myproject_new/runtime/logs/dsads_v4_refined_rerun_20261021

nohup bash /data/wanzhen/projects/myproject_new/scripts/run_dsads_v4_zsl.sh train-test \
  --gpu-id 1 --time-limit 300 --num-cpus 8 --resume \
  --attribute-vectors-root /data/wanzhen/projects/myproject_new/configs/attrIbutes_v4/dsads \
  --run-root /data/wanzhen/projects/myproject_new/runtime/zshar_runs \
  --run-name dsads_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42_refined_rerun_20261021 \
  > /data/wanzhen/projects/myproject_new/runtime/logs/dsads_v4_refined_rerun_20261021/train_test_gpu1.log 2>&1 < /dev/null &

echo $! > /data/wanzhen/projects/myproject_new/runtime/logs/dsads_v4_refined_rerun_20261021/train_test_gpu1.pid
```

默认分类器配置与上一轮一致：AutoGluon 默认模型集、`medium_quality`、加权集成、
bagging/stacking 为 0、每属性时限 300 秒、8 个 CPU。只以 seen 类训练，按 seed=42
的 20 组固定划分测试，每 split 为 6 类 × 30 个窗口。属性值为三级。

## 监控与结果

```bash
tail -F /data/wanzhen/projects/myproject_new/runtime/logs/dsads_v4_refined_rerun_20261021/train_test_gpu1.log
```

结果目录：

```text
/data/wanzhen/projects/myproject_new/runtime/zshar_runs/dsads/dsads_zsl_autogluon_default_ensemble_a2_v4_independent_t300_random20_eval30_seed42_refined_rerun_20261021/
```

除活动识别指标、每 split 的 Predictor 和预测结果外，还输出：

- `attribute_accuracy_summary.json`：先在 split 内对属性平均，再对 20 个 split 等权平均。
- `per_attribute_accuracy_summary.csv`：每个属性跨 split 的平均准确率、标准差和覆盖数。
- `split_attribute_accuracy.csv`：每个 split 的平均属性准确率。
- `attribute_accuracy_by_split.csv`：逐 split、逐属性的准确率。

逐属性均值只统计该属性出现的 split，不将未出现记为 0。全部 20 个 split 均完成后
汇总才标记为 `complete=true`。
