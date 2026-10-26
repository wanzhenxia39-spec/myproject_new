# 最新 initial9：PAMAP2 / MHEALTH GZSL，只有温度缩放

Python 入口：`run_initial9_gzsl_temperature_only.py`；后台入口：`run_initial9_gzsl_temperature_only.sh`。

- 原型与本体：`configs/attributes_new/`，包含 MHEALTH 最新 motion_regularity 修订。不修改通用数据集配置，不复用历史 ZSL 或旧 GZSL 分类器。
- 两个数据集分别读取 `split/gzsl_20/` 下同名 `*_gzsl_splits_seed42.json` 和 `*_gzsl_eval20_seen20_samples_seed42.json`。各 20 个划分，每划分 8 Seen + 4 Unseen，每类固定测试 20 个窗口（合计 240）。所有测试窗口从训练和校准数据中排除。
- 后端：每个属性独立 AutoGluon 默认候选模型 + `WeightedEnsemble`；`medium_quality`；每属性 300 秒预算；8 CPU / 1 GPU；bagging=0、stacking=0。原型匹配沿用 `ordinal_probability`，属性权重来自 Seen 验证集准确率。
- 校准：每个外层划分在 Seen 数据内建立 3 个 pseudo-GZSL 类别划分，每折 4 个 pseudo-Seen + 4 个 pseudo-Unseen，每类校准验证 3 个窗口。折内分类器只训练 pseudo-Seen 数据。按折平均验证 NLL 最小值选择温度，候选为 `0.01, 0.03, 0.1, 0.3, 1, 3, 10`。
- `gzsl_bias_correction=temperature_only`、`gamma=0`、`gamma_grid=None`，不搜索或应用 Seen penalty，不进行 AUSUC gamma 扫描。温度缩放在活动分数上进行，为 `softmax(score/T)`，只改变概率，不改变排序；相同模型下 Top-1、S/U/H 与无校准相同（忽略数值精度差异）。
- 需要重新训练主分类器及校准折内分类器，`train-test` 不是只加载历史模型的快速推理。`--resume` 只复用本次输出目录中协议与标签匹配的完整模型；中断模型目录保留为 `models.incomplete*` 后重新训练。
- 运行前检查真实路径与文件系统。模型、指标、预测、日志、缓存、输入快照和代码测试临时结果均位于 `/data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/`。

## 检查（不训练、不预测）

```bash
bash /home/wanzhen/myproject_new/scripts/run_initial9_gzsl_temperature_only.sh pamap2 0 check
bash /home/wanzhen/myproject_new/scripts/run_initial9_gzsl_temperature_only.sh mhealth 1 check
```

## 后台运行

入口在对应 GPU 上连续两次（间隔 30 秒）未发现计算进程后启动；已有任务继续运行。锁防止本入口重复启动同一数据集或抢占同一 GPU。

```bash
mkdir -p /data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/logs

nohup bash /home/wanzhen/myproject_new/scripts/run_initial9_gzsl_temperature_only.sh pamap2 0 train-test --resume \
  >> /data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/logs/pamap2_gpu0.log 2>&1 < /dev/null &
echo $! > /data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/logs/pamap2_gpu0.pid

nohup bash /home/wanzhen/myproject_new/scripts/run_initial9_gzsl_temperature_only.sh mhealth 1 train-test --resume \
  >> /data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/logs/mhealth_gpu1.log 2>&1 < /dev/null &
echo $! > /data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/logs/mhealth_gpu1.pid
```

每个数据集的结果目录为：

`/data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/runs/<dataset>/initial9_new_autogluon_default_temperature_only_cal3_t300_seed42/`

主要输出：`summary.csv`、`aggregate_metrics.csv/json`、`gzsl_per_split.csv`、`gzsl_report.csv/json`；各划分的 `metrics.json`、`details.csv`、`models/`、`calibration/`。报告包含 Seen accuracy、Unseen accuracy、H、overall accuracy、Top-1/Top-3、Macro-F1 的跨划分均值及样本标准差，还记录每划分温度和校准验证 NLL。

```bash
tail -f /data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/logs/pamap2_gpu0.log
tail -f /data/wanzhen/experiments/initial9_gzsl_temperature_only_seed42/logs/mhealth_gpu1.log
```
