# MHEALTH 新初始9维：seed42 ZSL、LOAO及四数据集差值图

入口：`run_mhealth_initial9_regularity.py`；GPU0包装脚本：`run_mhealth_initial9_regularity_gpu0.sh`。
脚本默认模式为 `check`，本次准备工作没有启动实际训练或推理。

## 输入与固定协议

- 新原型：`configs/attributes_new/activity_attribute_vectors_initial_mhealth_v1_5.json`。
- 保留原活动名、顺序及其他八维；只允许 `motion_regularity` 改值且不再为常量。
- 参考配置为原 `mhealth_zsl_autogluon_default_ensemble_initial_a1_v1_5_independent_t300_random20_eval30_seed42_gpu1_rerun/config.json`，不会改动或复用其模型。
- 固定 seed=42 的20个 split，每 split 4个Unseen类、每类固定30个原测试窗口；Seen数据按相同seed划分70%训练、30%验证。
- 后端 `autogluon`：9个独立属性预测器，默认模型集合，`medium_quality`，每属性300秒、8 CPU、1 GPU；bagging=0、stacking=0，开启weighted ensemble。
- 保留既有 CUDA LightGBM 与 NN_TORCH/FASTAI/CAT/XGB 的GPU配置；物理GPU0通过 `CUDA_VISIBLE_DEVICES=0` 映射到逻辑cuda:0。AutoGluon默认集合中的CPU模型仍按其原设置运行。
- 所有9个属性都在新运行目录重新训练；已有同协议新运行可用 `--resume` 恢复，旧常量regularity模型不能直接复用。
- 训练/选模只使用Seen数据；测试集不用于挑选属性、匹配权重或模型。测试候选仅为各split的Unseen类。

## 一条命令完成训练测试、LOAO和重绘

```bash
cd /home/wanzhen/myproject_new
bash scripts/run_mhealth_initial9_regularity_gpu0.sh check
nohup bash scripts/run_mhealth_initial9_regularity_gpu0.sh all --resume \
  > /data/wanzhen/experiments/mhealth_initial9_regularity_seed42/pipeline.log 2>&1 &
```

`check`校验新旧属性、固定窗口、原Seen训练划分和实际数据盘，不训练、不加载Predictor、不生成测试预测。
`all --resume`首次运行会训练全部新模型；后续仅复用该新运行内、原型哈希/训练目标/配置一致的模型，保留不完整模型目录后重新训练未完成的模型组。

也可分开执行：

```bash
bash scripts/run_mhealth_initial9_regularity_gpu0.sh train-test --resume
bash scripts/run_mhealth_initial9_regularity_gpu0.sh loao --resume
```

第二条会自动完成四数据集合并和绘图。已有完整LOAO指标时仅重画：

```bash
bash scripts/run_mhealth_initial9_regularity_gpu0.sh plot
```

包装脚本固定输出到下面的数据盘位置；直接使用Python入口可传 `--output-root /data/wanzhen/...`。禁止将输出指向系统盘或已迁移到系统盘的软链接。

## 输出

总目录：`/data/wanzhen/experiments/mhealth_initial9_regularity_seed42/`。

- `runs/mhealth/mhealth_zsl_autogluon_default_ensemble_initial9_regularity_v1_t300_random20_eval30_seed42_gpu0/`：新模型、各split测试预测、`summary.csv`、`aggregate_metrics.json`。
- `loao_4datasets/mhealth/split_*/`：本次新模型的属性概率、完整9维及九种删除配置的活动预测、指标、原型与权重。
- `loao_4datasets/loao_macro_f1_impact_4datasets.png/pdf/svg`：四数据集活动Macro-F1差值图。
- `loao_4datasets/loao_top1_impact_4datasets.png/pdf/svg`：四数据集Top-1差值图。
- `loao_4datasets/mhealth/loao_impact.png/pdf/svg`：MHEALTH单独的Top-1/Macro-F1双子图。
- `loao_4datasets/motion_regularity_effect.json`：regularity有/无的完整值、删除值、配对差值均值/标准差及正负split数。
- `loao_4datasets/motion_regularity_paired_20splits.csv`：该属性的20个split配对明细。
- `loao_4datasets/initial9_zsl_loao_results.xlsx`：完整值、删除值、9×4差值矩阵及逐split明细。
- `protocol.json`、`checks/`、`pipeline.log`、`cache/`：输入/分区溯源、预检查与测试记录、运行日志和缓存，全部位于数据盘。

四数据集列顺序固定为 PAMAP2、MHEALTH、USC-HAD、DSADS；WISDM不读取、不训练、不参与统计和绘图。
除MHEALTH外，其余三列读取系统盘 `attribute_prototype_analysis/new_results/initial9_zsl_loao/<dataset>/split_*/metrics.json` 的历史结果，不重新训练。
不覆盖原始五数据集历史图片。

## LOAO统计与解释

LOAO复用 `scripts/evaluate_initial9_zsl_loao.py:evaluate_split()`：先对新模型执行完整9维概率预测；九种消融只删除对应概率、原型与权重维度，按剩余权重和重新归一化序数概率匹配。每次删除不重新训练分类器，也不重新挑选模型或权重。

权重沿用同一批新模型在Seen验证集上的准确率。完整9维必须逐样本复现这批新ZSL模型保存的测试预测及Top-1/Macro-F1，否则停止。
Delta=完整9维−删除后8维，单位pp；20个配对split等权平均，误差线为样本标准差（ddof=1），不是置信区间。
这里Macro-F1是每个split内全部Unseen活动类别F1的宏平均，与单活动F1或一对其余二分类Macro-F1不同。

新模型在相同seed和时间预算下也不保证其他八个属性的最佳模型逐位不变，因此不能将新旧整体基线差异全部归因于regularity。
regularity的条件贡献应由同一批新模型的完整9维与删除regularity后的结果配对衡量；该差值可以为正、负或零。

## 代码位置

- 新训练/LOAO/四数据集汇总入口：`scripts/run_mhealth_initial9_regularity.py`。
- 原始初始9维LOAO入口与概率预测/匹配/绘图：`scripts/evaluate_initial9_zsl_loao.py`。
- 通用序数距离、删除维度与权重归一化：`src/zshar/evaluation/attribute_ablation.py`。
- 逐活动级汇总（本任务的四数据集总体Macro-F1图不调用它）：`/home/wanzhen/attribute_prototype_analysis/analyze_initial9_loao_by_activity.py`。

绘图函数已按传入数据集数量命名；原五数据集调用仍保留原文件名，四数据集调用生成4datasets和9x4文件名。
