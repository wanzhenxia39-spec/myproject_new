#!/usr/bin/env python3
"""Train reviewed MHEALTH initial9; LOAO; merge four datasets excluding WISDM."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts')]
DEFAULT_OUTPUT=Path('/data/wanzhen/experiments/mhealth_initial9_regularity_seed42')
REFERENCE=PROJECT/'runtime/zshar_runs/mhealth/mhealth_zsl_autogluon_default_ensemble_initial_a1_v1_5_independent_t300_random20_eval30_seed42_gpu1_rerun'
VECTORS=PROJECT/'configs/attributes_new/activity_attribute_vectors_initial_mhealth_v1_5.json'
HISTORICAL_LOAO=Path('/home/wanzhen/attribute_prototype_analysis/new_results/initial9_zsl_loao')
DATASETS=['pamap2','mhealth','uschad','dsads']
RUN_NAME='mhealth_zsl_autogluon_default_ensemble_initial9_regularity_v1_t300_random20_eval30_seed42_gpu0'
ATTR='motion_regularity'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalize(value):
    if isinstance(value,dict):return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,set):return sorted(normalize(v) for v in value)
    if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
    return str(value) if isinstance(value,Path) else value
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(normalize(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def data_output(path):
    path=Path(path).expanduser().resolve()
    if not path.is_relative_to(Path('/data/wanzhen').resolve()):
        raise ValueError('Training, prediction, metrics and logs must be under /data/wanzhen')
    parent=path
    while not parent.exists():parent=parent.parent
    if parent.stat().st_dev!=Path('/data').stat().st_dev or parent.stat().st_dev==Path('/').stat().st_dev:
        raise ValueError('Output must physically reside on the data filesystem, not a link to the system disk')
    path.mkdir(parents=True,exist_ok=True)
    return path


def build_config(base,reference,split_file,sample_file,indices,resume):
    from zshar.pipeline.runner import ExperimentConfig
    current=dict(reference)
    current.update(run_root=str(base/'runs'),run_name=RUN_NAME,
        split_manifest=str(split_file),sample_manifest=str(sample_file),split_indices=set(indices),
        reuse_models=resume,evaluate_only=False,dry_run=False)
    return ExperimentConfig(**current)


def assert_compatible_config(saved,current):
    ignored={'reuse_models','evaluate_only','dry_run'}
    a={k:v for k,v in normalize(saved).items() if k not in ignored}
    b={k:v for k,v in normalize(current).items() if k not in ignored}
    for value in [a,b]:
        if value.get('split_indices') is not None:value['split_indices']=sorted(value['split_indices'])
    if a!=b:raise ValueError('Existing run uses a different configuration; choose a new output root')


def validate_reannotation(old,new,attributes):
    if list(old)!=list(new) or len(new)!=12:raise ValueError('Expected identical 12 MHEALTH activities/order')
    if len(attributes)!=9 or ATTR not in attributes:raise ValueError('Expected initial 9-dimensional schema')
    if {r[ATTR] for r in old.values()}!={'High'}:raise ValueError('Historical regularity is no longer constant High')
    if len({r[ATTR] for r in new.values()})<2:raise ValueError('Reviewed regularity must vary across activities')
    for activity in old:
        if list(new[activity])!=attributes or any(v not in {'Low','Medium','High'} for v in new[activity].values()):
            raise ValueError('Invalid reviewed attribute schema or levels')
        if any(old[activity][a]!=new[activity][a] for a in attributes if a!=ATTR):
            raise ValueError('Only motion_regularity may differ from original initial9')


def prepare(base,resume):
    import attributes_v4_autogluon_common as common
    from zshar.data import load_activity_vectors,load_dataset_table
    from zshar.pipeline.training import split_train_validation
    from zshar.splits import load_sample_manifest,sample_evaluation_rows
    reference=read(REFERENCE/'config.json')
    required={'dataset':'mhealth','scenario':'zsl','backend':'autogluon','seed':42,
        'prototype_metric':'ordinal_probability','attribute_weighting':'validation_accuracy','samples_per_activity':30}
    if any(reference[k]!=v for k,v in required.items()):raise ValueError('Historical experiment protocol changed')
    expected_backend={'time_limit':300,'presets':'medium_quality','num_gpus':1,'num_cpus':8,
        'num_bag_folds':0,'num_stack_levels':0,'fit_weighted_ensemble':True,'model_suite':'default',
        'lightgbm_device':'cuda','gpu_model_types':['NN_TORCH','FASTAI','CAT','XGB'],'verbosity':2}
    if reference['backend_config']!=expected_backend:raise ValueError('Historical default ensemble settings changed')
    if any(reference.get(k) for k in ['attribute_vectors_root','attribute_feature_bindings','attribute_order_config']):
        raise ValueError('This rerun requires independent full-feature initial9 classifiers')
    spec,splits,split_file,sample_file=common.read_protocol('mhealth',seed=42)
    if set(reference['split_indices'])!={s.index for s in splits}:raise ValueError('Split indices changed')
    old,attributes=load_activity_vectors(spec.attributes_path)
    new,new_attributes=load_activity_vectors(VECTORS)
    if new_attributes!=attributes:raise ValueError('Reviewed schema differs from original')
    validate_reannotation(old,new,attributes)
    spec=replace(spec,attributes_path=VECTORS.resolve(),raw={**spec.raw,'attributes':str(VECTORS.resolve())})
    config=build_config(base,reference,split_file,sample_file,[s.index for s in splits],resume)
    run=Path(config.run_root)/'mhealth'/config.run_name
    if run.resolve()==REFERENCE.resolve():raise ValueError('Must not overwrite/reuse old MHEALTH models')
    if (run/'config.json').exists():assert_compatible_config(read(run/'config.json'),asdict(config))
    samples=load_sample_manifest(sample_file)
    table=load_dataset_table(spec)
    split_audits=[]
    for split in splits:
        evaluation=sample_evaluation_rows(table.frame,split,scenario='zsl',samples_per_activity=30,
                                         seed=42,manifest_ids=samples[split.index])
        if set(evaluation.window_index)!=samples[split.index] or len(evaluation)!=120 or not evaluation.activity_name.value_counts().eq(30).all():
            raise ValueError('Fixed 4 x 30 Unseen test samples differ from the original protocol')
        training=table.frame[table.frame.activity_name.isin(split.seen_activities)]
        fit,validation=split_train_validation(training,seed=42,validation_fraction=.3)
        if set(fit.window_index)&set(validation.window_index) or set(training.window_index)&samples[split.index]:
            raise ValueError('Seen fit/validation and Unseen evaluation must be disjoint')
        saved_manifest=read(REFERENCE/f'split_{split.index:03d}/models/attribute_backend_manifest.json')
        for attr in attributes:
            counts={str(k):int(v) for k,v in fit[attr].value_counts().items()}
            if attr!=ATTR and counts!=saved_manifest['class_counts'][attr]:
                raise ValueError('Unchanged attribute targets or original fit partition changed')
        split_audits.append({'split_index':split.index,'seen_fit_samples':len(fit),'seen_validation_samples':len(validation),
            'test_samples':len(evaluation),'regularity_fit_level_counts':{str(k):int(v) for k,v in fit[ATTR].value_counts().items()},
            'fit_window_ids_sha256':hashlib.sha256(fit.window_index.to_numpy(dtype='int64').tobytes()).hexdigest(),
            'validation_window_ids_sha256':hashlib.sha256(validation.window_index.to_numpy(dtype='int64').tobytes()).hexdigest()})
    inputs=[VECTORS,spec.attributes_path,PROJECT/'configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json',
            PROJECT/'configs/attributes/initial_attributes.json',split_file,sample_file,REFERENCE/'config.json']
    input_hashes={str(p.resolve()):sha(p) for p in inputs}
    feature_files={str(p.resolve()):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                   for p in [spec.features_path,spec.feature_names_path,spec.labels_path]}
    protocol={'experiment':'MHEALTH reviewed initial9 ZSL and four-dataset LOAO','physical_gpu':0,'seed':42,
        'datasets_in_plot':DATASETS,'mhealth_config':normalize(asdict(config)),
        'revised_vectors':str(VECTORS.resolve()),'revised_vectors_sha256':sha(VECTORS),
        'input_hashes':input_hashes,'feature_files':feature_files,'split_audits':split_audits,
        'loao':'Frozen full9 outputs; remove probability/prototype/weight coordinate; renormalize remaining weights',
        'delta':'full9 minus removed8; percentage points; SD over 20 paired splits',
        'test_selection':'none; classifiers/ensembles selected using Seen data only',
        'new_run':str(run),'loao_output':str(base/'loao_4datasets')}
    existing=base/'protocol.json'
    if existing.exists():
        previous=read(existing)
        assert_compatible_config(previous['mhealth_config'],protocol['mhealth_config'])
        for k in ['input_hashes','feature_files','split_audits','datasets_in_plot']:
            if previous[k]!=normalize(protocol[k]):raise ValueError('Saved run inputs or partitions changed')
    # Validate all pre-existing complete models against the NEW targets, never just directory names.
    for split in splits:
        model_dir=run/f'split_{split.index:03d}/models'
        if (model_dir/'attribute_backend_manifest.json').exists():
            verify_model_manifest(model_dir,table,split)
    return spec,config,run,splits,samples,table,protocol


def verify_model_manifest(model_dir,table,split):
    from zshar.pipeline.training import split_train_validation
    manifest=read(model_dir/'attribute_backend_manifest.json')
    if manifest['manager_type']!='independent' or manifest['classifier_backend']!='autogluon' or manifest['attribute_cols']!=table.attribute_columns:
        raise ValueError('Expected independent AutoGluon model manager with the reviewed schema')
    training=table.frame[table.frame.activity_name.isin(split.seen_activities)]
    fit,_=split_train_validation(training,seed=42,validation_fraction=.3)
    for attr in table.attribute_columns:
        counts={str(k):int(v) for k,v in fit[attr].value_counts().items()}
        if counts!=manifest['class_counts'][attr]:raise ValueError('Saved model targets do not match reviewed vectors')
    return manifest


def train_new(spec,config,run,protocol,resume):
    from run_mhealth_a1_autogluon_default_ensemble import preserve_partial_models
    from zshar.pipeline.runner import run_experiment
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Training requires physical GPU0 exposed as logical cuda:0')
    if (run/'config.json').exists() and not resume:
        raise FileExistsError('Run already exists; pass --resume or select a fresh --output-root')
    write(run/'reviewed_input_protocol.json',protocol)
    preserve_partial_models(run,resume=resume,dry_run=False)
    run_experiment(spec,config)
    print('TRAIN-TEST COMPLETE:',run,flush=True)


def require_completed_run(run,splits,table):
    from zshar.evaluation.metrics import classification_summary
    import pandas as pd
    for split in splits:
        directory=run/f'split_{split.index:03d}'
        saved=read(directory/'metrics.json')
        if saved['attribute_vectors']['sha256']!=sha(VECTORS):raise ValueError('Completed run used different prototypes')
        verify_model_manifest(directory/'models',table,split)
        details=pd.read_csv(directory/'details.csv')
        metric=classification_summary(details.activity_name,details.predicted_activity)
        if any(abs(metric[m]-saved['activity'][m])>1e-12 for m in ['accuracy','macro_f1']):
            raise ValueError('Saved full9 predictions and metrics disagree')


def validate_rows(rows,datasets):
    # Attribute names come from the ontology; metadata keys are never treated as variants.
    attrs=[a['name'] for a in read(PROJECT/'configs/attributes/initial_attributes.json')['attributes']]
    expected={'full9',*attrs}
    for ds in datasets:
        subset=[r for r in rows if r['dataset']==ds]
        indices={r['split_index'] for r in subset}
        if len(indices)!=20 or len(subset)!=200:raise ValueError(f'Need all 20 x 10 LOAO results for {ds}')
        for index in indices:
            group=[r for r in subset if r['split_index']==index]
            if len(group)!=10 or {r['variant'] for r in group}!=expected:raise ValueError('Missing/duplicate LOAO variants')
            baseline=next(r for r in group if r['variant']=='full9')
            for row in group:
                if abs(row['delta_top1']-(baseline['top1_accuracy']-row['top1_accuracy']))>1e-12:
                    raise ValueError('Incorrect paired accuracy delta')
                if abs(row['delta_macro_f1']-(baseline['macro_f1']-row['macro_f1']))>1e-12:
                    raise ValueError('Incorrect paired Macro-F1 delta')
    if {r['dataset'] for r in rows}!=set(datasets):raise ValueError('Unexpected dataset (WISDM must be excluded)')


def plot_four(base,protocol):
    import pandas as pd
    import evaluate_initial9_zsl_loao as old
    output=data_output(base/'loao_4datasets')
    rows=[];historical_hashes={}
    for ds in DATASETS:
        (output/ds).mkdir(exist_ok=True)
        source=output if ds=='mhealth' else HISTORICAL_LOAO
        paths=sorted((source/ds).glob('split_*/metrics.json'))
        if len(paths)!=20:raise ValueError(f'Need 20 completed split metrics: {source/ds}')
        for path in paths:
            saved=read(path)
            if ds=='mhealth' and saved['signature']['vectors_sha256']!=sha(VECTORS):
                raise ValueError('New MHEALTH LOAO was not computed using the reviewed prototypes')
            rows.extend(saved['rows'])
            if ds!='mhealth':historical_hashes[str(path.resolve())]=sha(path)
    validate_rows(rows,DATASETS)
    sys.path.insert(0,'/home/wanzhen/attribute_prototype_analysis/new_results/initial9_seen_predictability/cache/python_deps')
    old.summarize(output,rows,DATASETS)
    frame=pd.DataFrame(rows)
    focus=frame[(frame.dataset=='mhealth')&(frame.variant==ATTR)].copy()
    for metric in ['top1','macro_f1']:focus['delta_'+metric+'_pp']=100*focus['delta_'+metric]
    focus.to_csv(output/'motion_regularity_paired_20splits.csv',index=False)
    summary=pd.read_csv(output/'ablation_summary.csv')
    selected=summary[(summary.dataset=='mhealth')&(summary.attribute==ATTR)].iloc[0].to_dict()
    write(output/'motion_regularity_effect.json',selected)
    report=output/'REPORT_zh.md'
    report.write_text(report.read_text().replace('# 初始9维属性的逐属性移除实验',
        '# 初始9维属性逐维移除：新版 MHEALTH，四数据集（不含 WISDM）')+
        '\n\nMHEALTH 使用 configs/attributes_new 下重标注后的原型和本次重新训练的20个独立AutoGluon模型组；另外三个数据集沿用原 initial9_zsl_loao 的既有结果。MHEALTH 的完整9维与每个删除配置均来自同一批新模型，不能将新原型与旧常量regularity模型混用。\n\n'
        '训练配置、划分和seed保持原实验一致；按时间预算重新训练其他八维也可能改变其模型选择，不能将与历史基线的所有差异归因于regularity一维。应以同一批新模型的full9和删除regularity的配对差值评价该属性的条件贡献。\n\n'
        '重点结果见 motion_regularity_effect.json 和 motion_regularity_paired_20splits.csv。只有regularity存在/移除的对照与其余9个配置的匹配均不进行重新训练或测试调参。\n',encoding='utf-8')
    write(output/'protocol.json',{**protocol,'historical_other_dataset_metric_files':historical_hashes})
    print('MOTION_REGULARITY EFFECT:',json.dumps(selected,ensure_ascii=False),flush=True)
    print('FOUR-DATASET FIGURE:',output/'loao_macro_f1_impact_4datasets.png',flush=True)


def compute_loao(base,spec,run,splits,samples,table,resume):
    import torch
    import evaluate_initial9_zsl_loao as old
    torch.set_num_threads(4)
    require_completed_run(run,splits,table)
    output=data_output(base/'loao_4datasets')
    for split in splits:
        old.evaluate_split('mhealth',spec,table,run,split,samples[split.index],output,resume)
    print('LOAO COMPLETE: 20 splits x (full9 + 9 removals)',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check','train-test','loao','plot','all'],nargs='?',default='check')
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    base=data_output(args.output_root)
    for folder in ['tmp','pycache','matplotlib']:(base/'cache'/folder).mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(base/'cache/tmp');os.environ['MPLCONFIGDIR']=str(base/'cache/matplotlib')
    spec,config,run,splits,samples,table,protocol=prepare(base,args.resume)
    write(base/'protocol.json',protocol)
    print(json.dumps({'mode':args.mode,'physical_gpu':0,'seed':42,'vectors':str(VECTORS),'new_run':str(run),
        'split_count':20,'test_samples_per_activity':30,'backend_config':config.backend_config,
        'datasets_in_plot':DATASETS,'output':str(base)},ensure_ascii=False,indent=2),flush=True)
    if args.mode=='check':
        print('CHECK PASSED: no model training, loading, prediction, or LOAO test was started.',flush=True)
        return 0
    if args.mode in ['train-test','all']:train_new(spec,config,run,protocol,args.resume)
    if args.mode in ['loao','all']:compute_loao(base,spec,run,splits,samples,table,args.resume)
    if args.mode in ['loao','plot','all']:plot_four(base,protocol)
    return 0


if __name__=='__main__':raise SystemExit(main())
