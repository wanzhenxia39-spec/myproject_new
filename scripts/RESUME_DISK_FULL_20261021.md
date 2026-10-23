# 磁盘满后的三项实验续跑

所有结果保存在数据盘的 `/data/wanzhen/projects/myproject_new/runtime/zshar_runs`；
日志保存到 `/data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021`。
不要换 run-name 或清空原目录，否则不会复用之前完成的模型。

当前断点：DSADS 重新细化 10/20；WISDM A2 seed43 测试 4/20；DSADS 初始 A1 重跑 3/20。
两个训练入口支持 `--resume`：复用完成的模型，并保留失败 split 的残缺模型目录为
`models.incomplete_<时间戳>`，再重训该 split。每个已训练 split 会重新进行测试。
训练和测试均导出整体平均属性准确率及各属性跨 split 平均值。

## DSADS seed42 重新细化续跑

在终端中先设置 `NEW_API_KEY`。该任务不会训练分类器，不需要 GPU。

```bash
cd /data/wanzhen/projects/myproject_new
mkdir -p /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021
nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /data/wanzhen/projects/myproject_new/scripts/run_dsads_refinement_xxapi.py \
  --root /data/wanzhen/projects/myproject_new/configs/attrIbutes_v4/dsads \
  --stage all --model gpt-6.1-sol --temperature 0.7 \
  --proxy-mode direct --max-attempts 6 --retry-delay 60 \
  > /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021/dsads_refinement.log 2>&1 < /dev/null &
echo $! > /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021/dsads_refinement.pid
```

## WISDM A2 seed43，GPU0

```bash
cd /data/wanzhen/projects/myproject_new
mkdir -p /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021
nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /data/wanzhen/projects/myproject_new/scripts/run_wisdm_v4_autogluon_default_ensemble_seed43.py train-test \
  --gpu-id 0 --time-limit 300 --num-cpus 8 --resume \
  --attribute-vectors-root /data/wanzhen/projects/myproject_new/configs/attrIbutes_v4/wisdm/seed43 \
  --run-root /data/wanzhen/projects/myproject_new/runtime/zshar_runs \
  > /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021/wisdm_seed43_gpu0.log 2>&1 < /dev/null &
echo $! > /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021/wisdm_seed43_gpu0.pid
```

## DSADS 初始 A1 seed42 重跑，GPU1

```bash
cd /data/wanzhen/projects/myproject_new
mkdir -p /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021
nohup /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  /data/wanzhen/projects/myproject_new/scripts/run_mhealth_uschad_wisdm_dsads_initial9_autogluon_default.py train-test \
  --datasets dsads --gpu-id 1 --time-limit 300 --num-cpus 8 --resume \
  --split-root /data/wanzhen/projects/myproject_new/split/zero_shot_split \
  --run-root /data/wanzhen/projects/myproject_new/runtime/zshar_runs \
  --run-name-suffix rerun_20261021 \
  > /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021/dsads_initial9_gpu1.log 2>&1 < /dev/null &
echo $! > /data/wanzhen/projects/myproject_new/runtime/logs/resume_20261021/dsads_initial9_gpu1.pid
```

三条命令可并行运行。WISDM 和 DSADS 初始 A1 均使用 AutoGluon 默认集成，
每属性 300 秒、8 CPU、bagging=0、stacking=0、weighted ensemble 启用。
物理 GPU 变更不会改变 split、seed 或样本配置。

迁移前打开的终端可能仍持有已删除的旧工作目录，即使提示符显示 `~/myproject_new`。
务必先执行 `cd /data/wanzhen/projects/myproject_new`。以上脚本路径均使用绝对路径。
