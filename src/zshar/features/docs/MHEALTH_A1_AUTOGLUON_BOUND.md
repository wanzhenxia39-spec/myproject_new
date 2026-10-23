# MHEALTH 九维 A1：AutoGluon 默认集成与绑定特征

使用独立属性后端 `autogluon`：每个属性单独训练一个 AutoGluon predictor，
候选模型来自默认模型库，启用加权集成。逐属性输入由
`configs/attributes/initial_attributes_group.json` 决定，包含主组和辅助组。
不启用属性分类器链。

实验沿用 13 后端基准的 A1 v1.5 文件、seed 42、20 个活动划分、每划分
4 个 unseen 活动 × 30 个固定测试窗口、seen 内部 70%/30% 训练/验证、
`ordinal_probability` 匹配和 `validation_accuracy` 属性权重。
AutoGluon 的内部模型选择使用其训练数据内的默认留出策略，外部 seen
验证集用于属性评估和匹配权重，不加入 AutoGluon 训练。

默认参数为 `medium_quality`、每属性 600 秒、8 CPU、1 GPU、
`num_bag_folds=0`、`num_stack_levels=0`、`fit_weighted_ensemble=true`。
LightGBM 使用 CUDA，NN_TORCH、FASTAI、CAT、XGB 显式申请 GPU。
CPU 模型仍在 CPU 上执行；时间预算可能使部分候选模型来不及训练。
某属性在 seen 训练中只有一个等级时，沿用常量预测器。

实际属性输入维数依次为：motion_intensity 142、cadence 67、smoothness 15、
impulsiveness 15、posture_inclination 12、posture_variability 3、
rotational_motion 8、translational_motion 4、motion_regularity 140。
绑定在训练、验证、测试、概率预测和模型重载后均保持一致。续跑或独立测试
时发现已保存模型的特征绑定不同会报错，避免复用完整特征模型。

## 检查配置

```bash
cd /home/wanzhen/myproject_new
PYTHONPATH="$PWD/src" /home/wanzhen/miniconda3/envs/zara/bin/python \
  scripts/run_mhealth_a1_backend_benchmark.py check \
  --backends autogluon --feature-subsets --gpu-id 0
```

## GPU 0 后台训练并测试全部 20 个划分

```bash
cd /home/wanzhen/myproject_new
mkdir -p runtime/logs
logfile="runtime/logs/mhealth_a1_autogluon_bound_gpu0_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=0 \
  /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_mhealth_a1_backend_benchmark.py train-test \
  --backends autogluon --feature-subsets --gpu-id 0 --num-cpus 8 \
  --autogluon-time-limit 600 --autogluon-bag-folds 0 \
  --resume --fail-fast > "$logfile" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$logfile"
```

`tail -f "$logfile"` 查看进度。中断后重用相同命令即可续跑：完整模型会复用，
未完成的 split 模型目录会先归档再训练。每个划分训练后立即测试，最后汇总。

## 已训练模型单独重测

在上述训练完成后执行：

```bash
cd /home/wanzhen/myproject_new
mkdir -p runtime/logs
nohup env PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=0 \
  /home/wanzhen/miniconda3/envs/zara/bin/python -u \
  scripts/run_mhealth_a1_backend_benchmark.py evaluate \
  --backends autogluon --feature-subsets --gpu-id 0 --num-cpus 8 \
  --autogluon-time-limit 600 --autogluon-bag-folds 0 \
  --fail-fast > runtime/logs/mhealth_a1_autogluon_bound_gpu0_evaluate.log 2>&1 < /dev/null &
```

## 输出

- 模型、逐划分指标和 `summary.csv`：
  `runtime/zshar_runs/mhealth/mhealth_zsl_autogluon_a1_v1_5_bound_feature_subsets_random20_eval30_seed42/`
- 汇总 `comparison.csv`：
  `runtime/zshar_runs/mhealth/mhealth_a1_v1_5_autogluon_bound_feature_subsets_seed42/`
- 协议与原型副本：
  `runtime/mhealth_a1_autogluon_bound_feature_subsets_seed42/`

AutoGluon 的协议和汇总目录独立于已经完成的 13 后端结果。
原有基准脚本不指定 `--backends` 时，仍默认运行原来的 13 种后端。
如需调整训练预算或 bagging，使用对应参数并指定新的 `--run-root`、
`--protocol-root`、`--comparison-root`，避免与既有结果混用。
