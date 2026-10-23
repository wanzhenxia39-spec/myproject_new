# PAMAP2 / MHEALTH：v4 属性向量，13 个非链式后端，seed=42

入口：`run_v4_nonchain_backend_benchmark.py`；便捷启动器：
`run_pamap2_mhealth_v4_nonchain.sh`。默认 PAMAP2 使用物理 GPU 0，MHEALTH 使用物理 GPU 1。
每个数据集内部按后端串行执行，两个数据集可以同时启动。

使用 `configs/attrIbutes_v4/{pamap2,mhealth}/split_<原始编号>/` 中的最终 A2 向量，
验证细化方案、完整生成记录和向量一致性。使用 `split/zero_shot_split/` 中的
`*_unseen4_random20_splits_seed42.json` 和配套 `*_samples_seed42.json`。
全部 20 个划分都训练、测试；PAMAP2 每个划分测试 4×50=200 个未见类固定窗口，
MHEALTH 为 4×30=120 个。训练和验证只使用各划分的 seen 类，验证比例沿用项目默认 30%。
活动匹配为 `ordinal_probability`，属性权重为 seen 类验证集上的 `validation_accuracy`。

默认后端：`lightgbm,xgboost,catboost,logistic_regression,linear_svm,extra_trees,random_forest,neuralnet_fastai,neuralnet_torch,ordinal_logistic,coral_mlp,mlp_independent,mlp_multitask`。
`mlp_multitask` 共享特征主干，各属性输出头不传递预测结果，因此属于非链式模型。
排除 AutoGluon 默认集成，以及 multilabel/soft-OOF/classifier-chain 等所有链式后端。
FastAI、Torch 后端使用项目的原生实现。树提升和神经网络使用指定 GPU；
Logistic Regression、Linear SVM、Extra Trees、Random Forest 使用 CPU，每个任务默认 8 个线程。
训练超参数沿用各后端默认值；Linear SVM 开启概率校准以支持活动匹配。

所有模型、预测明细、指标、输入哈希快照、后台日志、临时文件和缓存统一保存到：

```text
/data/wanzhen/experiments/pamap2_mhealth_v4_nonchain_seed42/
  pamap2/ 或 mhealth/
    <dataset>_zsl_<backend>_a2_v4_nonchain_random20_eval<N>_seed42/
      split_<编号>/models/        模型及检查点
      split_<编号>/details.csv    测试预测明细
      split_<编号>/metrics.json   测试指标
      summary.csv                全部划分指标
      *attribute_accuracy*        按属性、按划分准确率汇总
      v4_input_manifest.json      向量、数据和划分配置的 SHA256
    logs/<backend>.log
    comparison/comparison.csv
    comparison/all_split_metrics.csv
    comparison/status.json
    runtime/                     临时文件及缓存
  nohup_logs/
```

输出路径会解析符号链接，必须在 `/data/wanzhen/` 内，且设备号须等于 `/data`、
不同于系统根目录。子进程工作目录也设在数据盘，以收纳库产生的相对路径文件。
`--output-root` 可以指定其他数据盘目录；系统盘路径会直接拒绝。

先做无训练检查：

```bash
cd /home/wanzhen/myproject_new
bash scripts/run_pamap2_mhealth_v4_nonchain.sh pamap2 check
bash scripts/run_pamap2_mhealth_v4_nonchain.sh mhealth check
```

启动两个独立后台任务：

```bash
cd /home/wanzhen/myproject_new
mkdir -p /data/wanzhen/experiments/pamap2_mhealth_v4_nonchain_seed42/nohup_logs
stamp=$(date +%Y%m%d_%H%M%S)

nohup bash scripts/run_pamap2_mhealth_v4_nonchain.sh pamap2 train-test --gpu-id 0 --num-cpus 8 --resume > "/data/wanzhen/experiments/pamap2_mhealth_v4_nonchain_seed42/nohup_logs/pamap2_gpu0_${stamp}.log" 2>&1 < /dev/null &
echo "PAMAP2 PID=$!"

nohup bash scripts/run_pamap2_mhealth_v4_nonchain.sh mhealth train-test --gpu-id 1 --num-cpus 8 --resume > "/data/wanzhen/experiments/pamap2_mhealth_v4_nonchain_seed42/nohup_logs/mhealth_gpu1_${stamp}.log" 2>&1 < /dev/null &
echo "MHEALTH PID=$!"
```

`--resume` 校验配置和输入快照后复用完整的模型；已完成 20 个划分的后端会跳过。
中断时尚未写完的模型目录会保留为 `models.incomplete_<时间戳>`，随后重训该划分。
单个后端失败会记录日志并继续后续后端；存在失败时进程以非零状态退出。
修复后可执行相同 nohup 命令续跑；也可用 `--backends lightgbm` 等仅运行指定后端。
如向量或协议变化，必须使用新的 `--output-root`，避免混用旧模型。

仅使用已训练模型重测：将 `train-test` 改为 `evaluate`。仅重新汇总：改为 `report`。
`evaluate` 缺模型时失败，不会隐式训练。`report` 仍会核对当前输入是否匹配训练快照。
检查模式只生成数据盘上的运行缓存目录，不训练或测试分类器。
