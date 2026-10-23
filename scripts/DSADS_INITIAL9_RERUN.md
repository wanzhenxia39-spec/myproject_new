# DSADS 初始 A1 v1.5：AutoGluon 默认集成独立重跑

使用 `configs/attributes/activity_attribute_vectors_initial_dsads_v1_5.json`：
新版本初始属性集，共 9 维，Low/Medium/High 三级值。

已存在的旧实验为
`/data/wanzhen/LLMbasedZeroshotHARAgent/zshar_runs/dsads/dsads_zsl_autogluon_default_ensemble_initial9_independent_t300_random20_eval30_seed42/`。
旧实验完成 20 个 split，Top-1 55.25%，按逐 split 属性测试准确率重新计算的平均属性准确率
为 63.30%。其向量哈希与当前 A1 文件一致；旧目录下 seed=42 的两份 split/sample JSON
与 `myproject_new/split/zero_shot_split` 下的文件字节一致。

原入口增加了 `--datasets dsads`、`--split-root` 和 `--run-name-suffix`。
运行下面命令会只重跑 DSADS，将结果保存到新目录；每次新实验指定不同后缀。

```bash
cd /home/wanzhen/myproject_new
mkdir -p runtime/logs/dsads_initial9_rerun
nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_mhealth_uschad_wisdm_dsads_initial9_autogluon_default.py train-test \
  --datasets dsads --gpu-id 0 --time-limit 300 --num-cpus 8 \
  --split-root /home/wanzhen/myproject_new/split/zero_shot_split \
  --run-root /home/wanzhen/myproject_new/runtime/zshar_runs \
  --run-name-suffix rerun_20261021 \
  > runtime/logs/dsads_initial9_rerun/rerun_20261021_gpu0.log 2>&1 < /dev/null &
echo $! > runtime/logs/dsads_initial9_rerun/rerun_20261021_gpu0.pid
```

固定 seed=42，20 个 split；每 split 6 未见类 × 每类 30 固定样本。
AutoGluon 默认模型集、medium_quality、weighted ensemble、bagging=0、stacking=0。
每属性 300 秒，8 CPU，物理 GPU0；默认从头训练，不加 `--resume`。
原型匹配与 v4 实验一致，为 ordinal_probability + validation_accuracy。

上述结果目录：

```text
runtime/zshar_runs/dsads/dsads_zsl_autogluon_default_ensemble_initial9_independent_t300_random20_eval30_seed42_rerun_20261021/
```

输出活动识别指标以及平均属性准确率、各属性跨 split 平均值、逐 split 属性均值。
整体先对 split 内属性求均值，再对 20 个 split 等权平均。
`initial9_input_snapshot.json` 记录 A1、ontology、split、测试窗口、数据及特征文件哈希。
同目录有训练锁；输入变化或无快照的旧模型不能直接复用。

只读检查：将 `train-test` 改为 `check`，并使用相同参数。
监控：`tail -f runtime/logs/dsads_initial9_rerun/rerun_20261021_gpu0.log`。
