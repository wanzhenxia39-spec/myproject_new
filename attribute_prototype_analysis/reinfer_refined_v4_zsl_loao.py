#!/usr/bin/env python3
"""Fresh-model-inference LOAO of split-specific A2 v4 ZSL prototypes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/home/wanzhen/myproject_new')
sys.path[:0] = [str(ROOT), str(PROJECT/'scripts'), str(PROJECT/'src')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
import pandas as pd
import attributes_v4_autogluon_common as common
from reinfer_initial9_zsl_loao import fresh_attribute_probabilities, save_array
from zshar.data import load_dataset_table, load_activity_vectors
from zshar.evaluation.attribute_ablation import ordinal_distance_components, ordinal_scores, candidate_predictions
from zshar.evaluation.metrics import classification_summary
from zshar.evaluation.scoring import validation_attribute_weights
from zshar.semantics.prototypes import rank_activity_prototypes
from zshar.splits import load_sample_manifest, sample_evaluation_rows

DATASETS = ['pamap2', 'mhealth', 'uschad', 'wisdm', 'dsads']
NAMES = dict(zip(DATASETS, ['PAMAP2', 'MHEALTH', 'USC-HAD', 'WISDM', 'DSADS']))
LEVELS = ['Low', 'Medium', 'High']
INITIAL = ['motion_intensity', 'cadence', 'smoothness', 'impulsiveness', 'posture_inclination',
           'posture_variability', 'rotational_motion', 'translational_motion', 'motion_regularity']
OUTPUT = Path('/data/wanzhen/experiments/attribute_ablation/refined_v4_zsl_loao_seed42')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def semantic_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(Path('/data/wanzhen').resolve()):
        raise ValueError('Model inference outputs must be under /data/wanzhen')
    parent = path
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path('/data').stat().st_dev or parent.stat().st_dev == Path('/').stat().st_dev:
        raise ValueError('Output must physically reside on the data filesystem')
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_path(ds):
    name = common.default_run_name(ds, 300)
    if ds == 'dsads':
        name += '_refined_rerun_20261021'
    return Path('/data/wanzhen/projects/myproject_new/runtime/zshar_runs')/ds/name


def audit_inputs():
    audit = {}
    for ds in DATASETS:
        spec, splits, split_file, sample_file = common.read_protocol(ds)
        vector_root = PROJECT/'configs/attrIbutes_v4'/ds
        vectors, errors = common.scan_vectors(vector_root, spec=spec, splits=splits)
        if errors:
            raise ValueError(errors)
        run = run_path(ds)
        config = common.read_json(run/'config.json')
        if any(config[k] != v for k, v in {'prototype_metric':'ordinal_probability',
                'attribute_weighting':'validation_accuracy', 'backend':'autogluon', 'scenario':'zsl'}.items()):
            raise ValueError('Unexpected original matching/training protocol')
        for split in splits:
            directory = run/f'split_{split.index:03d}'
            metric = common.read_json(directory/'metrics.json')
            manifest = common.read_json(directory/'models/attribute_backend_manifest.json')
            vector = vectors[split.index]
            if metric['attribute_vectors']['sha256'] != vector['sha256']:
                raise ValueError(f'Latest vector does not match trained models: {ds}/{split.index}')
            if manifest['attribute_cols'] != vector['attributes']:
                raise ValueError('Saved model schema differs from latest A2 schema')
            for semantic in vector['semantic_definitions'].values():
                if set(semantic['levels']) != set(LEVELS):
                    raise ValueError('This experiment expects ordered Low/Medium/High attributes')
            vector['model_manifest_sha256'] = common.digest(directory/'models/attribute_backend_manifest.json')
        audit[ds] = {'run':str(run), 'config_sha256':common.digest(run/'config.json'),
            'split_manifest':str(split_file), 'split_manifest_sha256':common.digest(split_file),
            'sample_manifest':str(sample_file), 'sample_manifest_sha256':common.digest(sample_file),
            'generation_config_sha256':common.digest(vector_root/'experiment_config.json'),
            'vectors':vectors}
    return audit


def verify_original_scores(probability, orders, prototypes, selected, candidates, weights, vectors, scores):
    """Check optimized matching against the project's original matcher, including drops."""
    for i in range(min(3, len(probability))):
        distributions = {a:{LEVELS[k]:float(probability[i,j,k]) for k in orders[j]}
                         for j,a in enumerate(selected)}
        distributions = {a:{k:v/sum(q.values()) for k,v in q.items()} for a,q in distributions.items()}
        ranking = rank_activity_prototypes(predicted_values={}, activity_vectors={c:vectors[c] for c in candidates},
            attribute_cols=selected, metric='ordinal_probability', attribute_weights=weights,
            probability_distributions=distributions)
        expected = dict((c,m.score) for c,m in ranking)
        if not np.allclose(scores[i], [expected[c] for c in candidates], atol=1e-12, rtol=0):
            raise ValueError('Vectorized ordinal matching differs from the original evaluator')


def evaluate_variant(directory, variant, selected, signature, model_dir, manifest, frame, features,
                     prototypes, candidates, weights, attributes, vectors, baseline_q, resume):
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory/'metrics.json'
    expected_signature = {**signature, 'variant':variant, 'selected_attributes':selected}
    if resume and result_file.exists():
        saved = common.read_json(result_file)
        if saved['signature'] != expected_signature:
            raise ValueError('Resume inputs changed')
        for filename, digest in saved['output_hashes'].items():
            if common.digest(directory/filename) != digest:
                raise ValueError('Completed output changed')
        with np.load(directory/'fresh_probabilities.npz') as p:
            q = p['probabilities']
        with np.load(directory/'activity_prediction.npz') as p:
            prediction = p['prediction']
        return saved, q, prediction
    if result_file.exists():
        raise FileExistsError(result_file)
    # Fresh load and predict for every retained classifier in EVERY configuration.
    q, orders, calls = fresh_attribute_probabilities(model_dir, manifest, frame, features, selected)
    indices = [attributes.index(a) for a in selected]
    components = ordinal_distance_components(q, prototypes[:, indices], orders)
    selected_weights = {a:weights[a] for a in selected}
    scores = ordinal_scores(components, list(selected_weights.values()))
    verify_original_scores(q, orders, prototypes[:,indices], selected, candidates, selected_weights, vectors, scores)
    prediction = candidate_predictions(scores, candidates)
    metrics = classification_summary(frame.activity_name, prediction)
    save_array(directory/'fresh_probabilities.npz', probabilities=q)
    save_array(directory/'activity_prediction.npz', prediction=prediction)
    exported = frame[['window_index','user_id','activity_id','activity_name']].copy()
    for j, a in enumerate(selected):
        exported[a+'__target'] = frame.activity_name.map(lambda c:vectors[c][a])
        for k, level in enumerate(LEVELS):
            exported[a+'__p_'+level] = q[:,j,k]
    exported['predicted_activity'] = prediction
    exported.to_csv(directory/'fresh_test_predictions.csv.gz', index=False, compression='gzip')
    common.write_json(directory/'inference_calls.json', calls)
    saved = {'signature':expected_signature, 'top1_accuracy':metrics['accuracy'], 'macro_f1':metrics['macro_f1'],
        'native_class_orders':orders, 'candidate_order':candidates, 'fresh_load_calls':len(calls),
        'fresh_predict_proba_calls':len(calls), 'autogluon_calls':sum(c['backend']=='autogluon' for c in calls),
        'constant_backend_calls':sum(c['backend']=='constant' for c in calls),
        'effective_weights':{a:w/sum(selected_weights.values()) for a,w in selected_weights.items()},
        'probabilities_identical_to_fresh_full':None if baseline_q is None else bool(np.array_equal(q,baseline_q[:,indices,:])),
        'probability_max_abs_difference_from_fresh_full':None if baseline_q is None else float(np.abs(q-baseline_q[:,indices,:]).max()),
        'output_hashes':{name:common.digest(directory/name) for name in [
            'fresh_probabilities.npz','activity_prediction.npz','inference_calls.json','fresh_test_predictions.csv.gz']}}
    common.write_json(result_file, saved)
    del components, scores
    gc.collect()
    return saved, q, prediction


def run_dataset(ds, output_string, audit, resume, threads):
    import torch
    torch.set_num_threads(threads)
    output = Path(output_string)
    spec, splits, _, sample_file = common.read_protocol(ds)
    samples = load_sample_manifest(sample_file)
    table = load_dataset_table(spec)
    run = Path(audit['run'])
    rows, checks = [], []
    for split in splits:
        started = time.monotonic()
        meta = audit['vectors'].get(split.index, audit['vectors'].get(str(split.index)))
        vectors, attributes = load_activity_vectors(meta['path'])
        if common.digest(meta['path']) != meta['sha256']:
            raise ValueError('Prototypes changed after audit')
        frame = sample_evaluation_rows(table.frame, split, scenario='zsl',
            samples_per_activity=common.EVAL_COUNTS[ds], seed=42, manifest_ids=samples[split.index])
        if set(frame.window_index) != samples[split.index] or not frame.activity_name.value_counts().eq(common.EVAL_COUNTS[ds]).all():
            raise ValueError('Test windows differ from the fixed original protocol')
        model_dir = run/f'split_{split.index:03d}/models'
        manifest = common.read_json(model_dir/'attribute_backend_manifest.json')
        if common.digest(model_dir/'attribute_backend_manifest.json') != meta['model_manifest_sha256']:
            raise ValueError('Saved model manifest changed')
        weights = validation_attribute_weights(attributes, manifest['validation_report'])
        candidates = list(split.unseen_activities)
        prototypes = np.array([[LEVELS.index(vectors[c][a]) for a in attributes] for c in candidates])
        signature = {'dataset':ds, 'split_index':int(split.index), 'vectors_sha256':meta['sha256'],
            'plan_sha256':meta['plan_sha256'], 'model_manifest_sha256':meta['model_manifest_sha256'],
            'test_window_indices':list(map(int,frame.window_index)),
            'feature_sha256':hashlib.sha256(np.ascontiguousarray(frame[table.feature_columns].to_numpy()).tobytes()).hexdigest(),
            'protocol':'fresh_model_load_and_predict_per_configuration_v1'}
        directory = output/ds/f'split_{split.index:03d}'
        directory.mkdir(parents=True,exist_ok=True)
        common.write_json(directory/'input_provenance.json', {**meta, 'weights':weights, 'candidates':candidates})
        baseline = baseline_q = None
        for variant in ['full'] + attributes:
            selected = attributes if variant=='full' else [a for a in attributes if a!=variant]
            saved,q,prediction = evaluate_variant(directory/variant, variant, selected, signature, model_dir,
                manifest, frame, table.feature_columns, prototypes, candidates, weights, attributes, vectors,
                baseline_q, resume)
            if variant=='full':
                baseline, baseline_q = saved, q
                old = pd.read_csv(run/f'split_{split.index:03d}/details.csv').set_index('window_index').loc[frame.window_index]
                historical = common.read_json(run/f'split_{split.index:03d}/metrics.json')['activity']
                if not np.array_equal(prediction,old.predicted_activity.astype(str).to_numpy()):
                    raise ValueError(f'Full A2 baseline does not reproduce historical predictions: {ds}/{split.index}')
                if not np.isclose(saved['top1_accuracy'],historical['accuracy'],atol=1e-12,rtol=0) or not np.isclose(saved['macro_f1'],historical['macro_f1'],atol=1e-12,rtol=0):
                    raise ValueError('Historical metrics differ')
            rows.append({'dataset':ds, 'split_index':int(split.index), 'removed_attribute':'' if variant=='full' else variant,
                'variant':variant, 'full_attribute_count':len(attributes), 'retained_attribute_count':len(selected),
                'samples':len(frame), 'top1_accuracy':saved['top1_accuracy'], 'macro_f1':saved['macro_f1'],
                'baseline_top1_accuracy':baseline['top1_accuracy'], 'baseline_macro_f1':baseline['macro_f1'],
                'delta_top1_pp':100*(baseline['top1_accuracy']-saved['top1_accuracy']),
                'delta_macro_f1_pp':100*(baseline['macro_f1']-saved['macro_f1']),
                'semantic_sha256':'' if variant=='full' else semantic_hash(meta['semantic_definitions'][variant])})
            checks.append({'dataset':ds, 'split_index':int(split.index), 'variant':variant,
                **{k:saved[k] for k in ['fresh_load_calls','fresh_predict_proba_calls','autogluon_calls',
                    'constant_backend_calls','probabilities_identical_to_fresh_full','probability_max_abs_difference_from_fresh_full']}})
        common.write_json(directory/'metrics.json', {'rows':rows[-(len(attributes)+1):]})
        print(f'DONE {ds} split={split.index} attrs={len(attributes)} fresh_calls={len(attributes)**2} seconds={time.monotonic()-started:.1f}',flush=True)
    return rows, checks


def aggregate_deltas(frame, keys):
    rows=[]
    for values, group in frame.groupby(keys,sort=False):
        if not isinstance(values,tuple): values=(values,)
        row=dict(zip(keys,values))
        row['n_splits']=int(group.split_index.nunique())
        for metric in ['top1','macro_f1']:
            x=group['delta_'+metric+'_pp']
            row[metric+'_delta_mean_pp']=float(x.mean())
            row[metric+'_delta_sd_pp']=float(x.std(ddof=1)) if len(x)>1 else np.nan
            row[metric+'_positive_splits']=int((x>1e-10).sum())
            row[metric+'_negative_splits']=int((x < -1e-10).sum())
            row[metric+'_zero_splits']=int((x.abs()<=1e-10).sum())
            row[metric+'_full_mean_pct']=float(100*group['baseline_'+('top1_accuracy' if metric=='top1' else 'macro_f1')].mean())
            row[metric+'_removed_mean_pct']=float(100*group['top1_accuracy' if metric=='top1' else 'macro_f1'].mean())
        row['semantic_variants']=int(group.semantic_sha256.nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def ordered(names):
    names=set(names)
    return [n for n in INITIAL if n in names]+sorted(names-set(INITIAL))


def plot_panels(frame, groups, metric, destination, title, show_n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    height=max(4.5,0.29*max(len(x) for x in groups.values())+1.6)
    fig,axes=plt.subplots(1,5,figsize=(23,height),squeeze=False)
    for ax,ds in zip(axes[0],DATASETS):
        names=groups[ds]
        g=frame[frame.dataset==ds].set_index('removed_attribute').loc[names]
        values=g[metric+'_delta_mean_pp'].to_numpy()
        errors=g[metric+'_delta_sd_pp'].to_numpy()
        y=np.arange(len(names))
        ax.barh(y,values,color=np.where(values < -1e-10,'#d88663','#6c9fd1'),height=.65)
        mask=np.isfinite(errors)
        ax.errorbar(values[mask],y[mask],xerr=errors[mask],fmt='none',color='#454545',capsize=2,lw=.8)
        ax.axvline(0,color='black',lw=.8)
        labels=[n.replace('_',' ') + (f' (n={int(g.loc[n,"n_splits"])})' if show_n else '') for n in names]
        ax.set_yticks(y,labels,fontsize=8)
        ax.invert_yaxis()
        ax.set_title(NAMES[ds],fontsize=12)
        ax.set_xlabel('Full minus removed (pp)',fontsize=9)
        ax.grid(axis='x',alpha=.2)
        for i,v in enumerate(values):
            ax.annotate(f'{v:+.2f}',(v,i),xytext=(3 if v>=0 else -3,0),textcoords='offset points',
                        ha='left' if v>=0 else 'right',va='center',fontsize=7)
        left,right=ax.get_xlim(); margin=(right-left)*.23+0.2
        ax.set_xlim(left-margin,right+margin)
    fig.suptitle(title+'\nPositive: removal hurts; negative: removal helps. Error bars: paired split SD.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:
        fig.savefig(destination.with_suffix('.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)


def summarize(output,rows,checks,audit):
    sys.path.insert(0,str(ROOT/'new_results/initial9_seen_predictability/cache/python_deps'))
    detail=pd.DataFrame(rows)
    detail.to_csv(output/'per_split_metrics.csv',index=False)
    check=pd.DataFrame(checks)
    check.to_csv(output/'inference_audit.csv',index=False)
    drops=detail[detail.variant!='full'].copy()
    summary=aggregate_deltas(drops,['dataset','removed_attribute'])
    exact=aggregate_deltas(drops,['dataset','removed_attribute','semantic_sha256'])
    summary.to_csv(output/'all_attributes_conditional_summary.csv',index=False)
    exact.to_csv(output/'exact_semantics_conditional_summary.csv',index=False)
    common_names=ordered(set.intersection(*(set(summary.loc[(summary.dataset==ds)&(summary.n_splits==20),'removed_attribute']) for ds in DATASETS)))
    common_summary=summary[summary.removed_attribute.isin(common_names)].copy()
    common_summary.to_csv(output/'common_attributes_summary.csv',index=False)
    baseline=detail[detail.variant=='full'].groupby('dataset').agg(
        n_splits=('split_index','nunique'), min_attributes=('full_attribute_count','min'),max_attributes=('full_attribute_count','max'),
        top1_mean=('top1_accuracy','mean'),top1_sd=('top1_accuracy','std'),
        macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reindex(DATASETS).reset_index()
    baseline.to_csv(output/'full_refined_baseline.csv',index=False)
    semantic_rows=[]
    for ds in DATASETS:
        for split, meta in audit[ds]['vectors'].items():
            for attr, semantic in meta['semantic_definitions'].items():
                semantic_rows.append({'dataset':ds,'split_index':int(split),'attribute':attr,
                    'semantic_sha256':semantic_hash(semantic), 'definition':semantic['definition'],
                    'Low':semantic['levels']['Low'],'Medium':semantic['levels']['Medium'],'High':semantic['levels']['High']})
    semantics=pd.DataFrame(semantic_rows)
    semantics.to_csv(output/'attribute_semantics_by_split.csv',index=False)
    # Keep different definitions separate, even when the model attribute name agrees.
    exact_plot=exact.copy()
    display=[]
    semantic_catalog=[]
    for row in exact.to_dict('records'):
        ds,attr,h=row['dataset'],row['removed_attribute'],row['semantic_sha256']
        occurrences=semantics[(semantics.dataset==ds)&(semantics.attribute==attr)&(semantics.semantic_sha256==h)]
        split_ids=sorted(occurrences.split_index.unique())
        variants=summary.loc[(summary.dataset==ds)&(summary.removed_attribute==attr),'semantic_variants'].iloc[0]
        label=attr if variants==1 else attr+' [s'+','.join(map(str,split_ids))+']'
        display.append(label)
        semantic_catalog.append({**row,'display_name':label,'split_indices':','.join(map(str,split_ids)),
                                 'definition':occurrences.definition.iloc[0]})
    exact_plot['removed_attribute']=display
    semantic_catalog=pd.DataFrame(semantic_catalog)
    semantic_catalog.to_csv(output/'exact_semantics_catalog.csv',index=False)
    # A common-name panel is only valid when the common attributes also share definitions.
    if not all(semantics[semantics.attribute==a].semantic_sha256.nunique()==1 for a in common_names):
        raise ValueError('Common-name attributes have different semantics; use exact-semantic panels')
    for metric,label in [('top1','Top-1 accuracy'),('macro_f1','Activity Macro-F1')]:
        groups={ds:common_names for ds in DATASETS}
        plot_panels(summary,groups,metric,output/f'loao_{metric}_common_impact_5datasets',
                    f'A2 v4 LOAO: {label}; attributes shared by all 100 splits (n=20 per dataset)',False)
        groups={ds:ordered(summary.loc[(summary.dataset==ds)&(summary.n_splits==20),'removed_attribute']) for ds in DATASETS}
        plot_panels(summary,groups,metric,output/f'loao_{metric}_dataset_common_impact_5datasets',
                    f'A2 v4 LOAO: {label}; dataset-specific common attributes (n=20)',False)
        groups={ds:ordered(summary.loc[summary.dataset==ds,'removed_attribute']) for ds in DATASETS}
        plot_panels(summary,groups,metric,output/f'loao_{metric}_all_attributes_impact_5datasets',
                    f'A2 v4 LOAO: {label}; same-name exploratory pooling (definitions may differ; n/20)',True)
        groups={ds:ordered(exact_plot.loc[exact_plot.dataset==ds,'removed_attribute']) for ds in DATASETS}
        plot_panels(exact_plot,groups,metric,output/f'loao_{metric}_exact_semantics_impact_5datasets',
                    f'A2 v4 LOAO: {label}; exact semantic groups (s: split IDs; n=1: no SD)',True)
        means=common_summary.pivot(index='removed_attribute',columns='dataset',values=metric+'_delta_mean_pp').reindex(index=common_names,columns=DATASETS)
        means.to_csv(output/f'{metric}_common_delta_mean_pp_4x5.csv')
    with pd.ExcelWriter(output/'refined_v4_loao_results.xlsx',engine='openpyxl') as writer:
        for name,frame in [('common_attributes',common_summary),('all_attributes_conditional',summary),
                           ('exact_semantics',semantic_catalog),('full_baseline',baseline),('per_split',detail),
                           ('attribute_semantics',semantics),('inference_audit',check)]:
            frame.to_excel(writer,sheet_name=name,index=False)
            writer.sheets[name].freeze_panes='C2'
            writer.sheets[name].auto_filter.ref=writer.sheets[name].dimensions
    expected_calls=int(sum(len(v['attributes'])**2 for ds in audit.values() for v in ds['vectors'].values()))
    expected_configs=int(sum(len(v['attributes'])+1 for ds in audit.values() for v in ds['vectors'].values()))
    verification={'complete':len(detail)==expected_configs,'splits':int((detail.variant=='full').sum()),
        'configurations':len(detail),'expected_configurations':expected_configs,
        'fresh_model_load_calls':int(check.fresh_load_calls.sum()),'expected_fresh_calls':expected_calls,
        'fresh_predict_proba_calls':int(check.fresh_predict_proba_calls.sum()),
        'autogluon_calls':int(check.autogluon_calls.sum()),'constant_calls':int(check.constant_backend_calls.sum()),
        'historical_full_baselines_exactly_reproduced':100,'common_attributes':common_names,
        'drop_configurations_identical_probabilities_to_fresh_full':int(check.probabilities_identical_to_fresh_full.map(lambda x:str(x).lower()=='true').sum()),
        'drop_configurations':len(drops)}
    if not verification['complete'] or verification['fresh_model_load_calls']!=expected_calls:
        raise ValueError('Incomplete configurations or call counts')
    common.write_json(output/'verification.json',verification)
    lines=['# 最新版 A2 v4 细化属性逐维移除实验','',
        '差值定义：完整细化属性集 − 删除该属性后的结果，单位为百分点（pp）。正值表示删除导致下降，负值表示删除后改善。',
        '完整基线是每个 split 各自的完整 A2 属性集（9–12 维），不是先裁成公共维度。每次只从该完整基线移除一维，其他维度、验证准确率权重、候选类别和测试窗口固定；剩余权重重新归一化。',
        '本次沿用每个 split 已保存的独立属性模型，不重训、不重新选模；完整配置和每个删除配置都独立重新加载全部保留分类器并调用 predict_proba，概率和原型同时去除该维后重新执行序数概率匹配。',
        '使用 configs/attrIbutes_v4 下最新 full_a2_generation_v1 的原型；全部100份原型 SHA256 与对应已完成模型运行一致。DSADS 使用 refined_rerun_20261021；其余均使用 seed42 的 A2 v4 independent t300 运行。',
        '所有 split 使用固定 ZSL Unseen 测试窗口与仅 Unseen 的候选类别。Macro-F1 为活动分类指标。模型和权重在本次测试前已冻结。','',
        '## 不同 split 属性集不同：三个统计层次','',
        '1. 主图：所有100个 split 共同保留且定义与等级描述完全一致的属性：'+', '.join(common_names)+'。每个数据集每个条形均基于20个完整配对差值。',
        '2. 数据集公共属性图：各数据集自身20个 split 的交集。可评价在本数据集稳定保留、但不属于全局交集的属性。',
        '3. 精确定义补充图（exact_semantics）：所有实际出现的属性按“名称＋定义＋三级等级描述”精确分组。定义变化时不混合平均，标签 s 标明对应 split；标注 n，未出现不填0，n=1 不计算标准差。新增或拆分属性的不同名称不强行合并。',
        '同名合并图（all_attributes）仅作为探索性索引，可能混合不同定义，不能作为属性必要性的主证据。定义变化数见 semantic_variants，精确定义及 split 清单见 exact_semantics_catalog.csv。条件子集的均值只能解释该子集，不能与20个 split 的均值直接作为公平排名；不同完整属性上下文也可能影响删除贡献。',
        '误差线为各 split 配对差值的样本标准差（ddof=1），不是置信区间。20个 split 共享数据，不能据此声称统计显著或独立重复。测试消融属于事后分析，不应再用这些测试结果选择部署属性子集。','',
        '## 完整细化基线（20 splits）','',
        '|数据集|维数|Top-1 (%)|Macro-F1 (%)|','|---|---|---:|---:|']
    for row in baseline.to_dict('records'):
        lines.append(f'|{NAMES[row["dataset"]]}|{row["min_attributes"]}–{row["max_attributes"]}|{100*row["top1_mean"]:.2f} ± {100*row["top1_sd"]:.2f}|{100*row["macro_f1_mean"]:.2f} ± {100*row["macro_f1_sd"]:.2f}|')
    for metric,label in [('top1','Top-1'),('macro_f1','Macro-F1')]:
        lines+=['',f'## {label}：全局公共属性差值 (pp)','',
                '|属性|'+'|'.join(NAMES[ds] for ds in DATASETS)+'|', '|---|'+'---:|'*5]
        for attr in common_names:
            values=[float(common_summary.loc[(common_summary.dataset==ds)&(common_summary.removed_attribute==attr),metric+'_delta_mean_pp'].iloc[0]) for ds in DATASETS]
            lines.append('|'+attr+'|'+'|'.join(f'{v:+.3f}' for v in values)+'|')
        shared_negative=common_summary[common_summary[metric+'_delta_mean_pp'] < -1e-10]
        stable=summary[summary.n_splits==20]
        stable_negative=stable[stable[metric+'_delta_mean_pp'] < -1e-10]
        lines+=['',f'全局公共属性共 {len(common_summary)} 个属性×数据集组合，其中负均值 {len(shared_negative)} 个；各数据集自身20个 split 公共属性共 {len(stable)} 组，其中负均值 {len(stable_negative)} 个。这两层均有完整20个配对结果，是本次主要分析依据。']
        negative=summary[summary[metric+'_delta_mean_pp'] < -1e-10]
        lines+=['',f'探索性同名属性×数据集条件均值：{len(summary)} 项中，负值 {len(negative)} 项。以下仅作检索（含实际出现 split 数），不同定义混合的组不用于必要性结论：','',
                '|数据集|属性|n/20|差值 pp|','|---|---|---:|---:|']
        for r in negative.to_dict('records'):
            lines.append(f'|{NAMES[r["dataset"]]}|{r["removed_attribute"]}|{r["n_splits"]}|{r[metric+"_delta_mean_pp"]:+.3f}|')
    lines+=['','## 解释边界','',
        '负均值表示在当前冻结分类器、原型、匹配权重及测试划分下，该属性的加入使对应平均指标低于单维删除后；零均值也可能由正负效果抵消，需查看逐 split 明细。不能由一个负均值断言属性的语义设计无用，也不能将多个单维删除收益相加。',
        '仅统计全局交集会漏掉细化引入的属性，因此主图与完整补充图一起解读。新增属性仅在部分 split 出现，样本较少，不能据其条件均值推出对全部 split 的普遍必要性。',
        '', '## 完整性核对','', '```json',json.dumps(verification,ensure_ascii=False,indent=2),'```']
    (output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    index=['# A2 v4 LOAO 结果索引','',
        '推荐先看主图：四个全局公共属性，在五个数据集各20个 split 上的配对差值。完整基线仍使用各 split 自己的完整细化属性集。','',
        '- [Top-1 主图](loao_top1_common_impact_5datasets.png) · [PDF](loao_top1_common_impact_5datasets.pdf)',
        '- [活动 Macro-F1 主图](loao_macro_f1_common_impact_5datasets.png) · [PDF](loao_macro_f1_common_impact_5datasets.pdf)',
        '- [Excel 结果表](refined_v4_loao_results.xlsx)',
        '- [中文分析与协议说明](REPORT_zh.md)','',
        '## 补充结果','',
        '- dataset_common：每个数据集各自20个 split 均存在的属性，优先于稀疏条件统计。',
        '- exact_semantics：按名称、定义和等级描述精确分组的全部属性图；标签 s 为对应 split，n=1 无标准差。',
        '- all_attributes：同名属性合并的探索性索引，可能混合不同语义定义，不能作为必要性的主证据。',
        '- exact_semantics_catalog.csv：每个精确定义组的 split 清单、定义、指标差值与标准差。',
        '- per_split_metrics.csv：所有1105个完整/移除配置的绝对值和配对差值。',
        '- input_audit.json：最新版原型、生成计划及对应模型运行的哈希；各配置保存实际重载预测记录、概率和活动预测。',
        '- verification.json / artifact_verification.json：配置计数及对保存概率、预测和指标的复核。','',
        '差值 = 完整 − 删除，单位 pp；误差线为20个或 n 个配对差值的标准差，不是置信区间。','']
    (output/'RESULT_INDEX_zh.md').write_text('\n'.join(index),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    output=validate_output(args.output)
    audit=audit_inputs()
    audit_file=output/'input_audit.json'
    if audit_file.exists() and canonical(common.read_json(audit_file))!=canonical(common.normalize(audit)):
        raise ValueError('Input audit differs from previous invocation')
    common.write_json(audit_file,audit)
    print('AUDIT: 100 latest A2 v4 prototypes match model runs and generation provenance',flush=True)
    if args.audit_only:return
    snapshot=output/'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    sources=[Path(__file__), ROOT/'reinfer_initial9_zsl_loao.py',ROOT/'test_refined_v4_zsl_loao.py',
             ROOT/'verify_refined_v4_zsl_loao.py',
             ROOT/'run_refined_v4_zsl_loao.sh',
             PROJECT/'src/zshar/evaluation/attribute_ablation.py',PROJECT/'scripts/attributes_v4_autogluon_common.py']
    for source in sources:shutil.copy2(source,snapshot/source.name)
    common.write_json(snapshot/'sha256.json',{str(p):common.digest(p) for p in sources})
    if args.summarize_only:
        rows=pd.read_csv(output/'per_split_metrics.csv',keep_default_na=False).to_dict('records')
        checks=pd.read_csv(output/'inference_audit.csv').to_dict('records')
        summarize(output,rows,checks,audit)
        return
    rows=[];checks=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_dataset,ds,str(output),audit[ds],args.resume,args.threads):ds for ds in DATASETS}
        for future in as_completed(futures):
            r,c=future.result();rows.extend(r);checks.extend(c)
    summarize(output,rows,checks,audit)


if __name__=='__main__':main()
