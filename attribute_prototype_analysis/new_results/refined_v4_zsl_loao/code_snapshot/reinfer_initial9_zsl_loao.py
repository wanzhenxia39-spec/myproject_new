#!/usr/bin/env python3
"""Reload and predict every retained classifier independently for each LOAO variant."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parent
PROJECT = Path('/home/wanzhen/myproject_new')
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(PROJECT / 'src'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import pandas as pd

import evaluate_initial9_zsl_loao as previous
from zshar.data import load_dataset_table
from zshar.evaluation.attribute_ablation import candidate_predictions, ordinal_distance_components, ordinal_scores
from zshar.evaluation.metrics import classification_summary
from zshar.evaluation.scoring import validation_attribute_weights
from zshar.splits import sample_evaluation_rows

OUTPUT = CODE_ROOT / 'new_results/initial9_zsl_loao_new'
OLD = CODE_ROOT / 'new_results/initial9_zsl_loao'
LEVELS = previous.LEVELS


def validate_system_output(path):
    path = Path(path).expanduser().resolve()
    if path != OUTPUT.resolve():
        raise ValueError(f'This experiment must write only to {OUTPUT}')
    parent = path
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path('/').stat().st_dev or parent.stat().st_dev == Path('/data').stat().st_dev:
        raise ValueError('Expected the actual system filesystem')
    path.mkdir(parents=True, exist_ok=True)
    return path


def fresh_attribute_probabilities(model_dir, manifest, frame, feature_columns, attributes, loader=None):
    """No predictor or probability cache: a fresh loader call for every selected attribute."""
    if loader is None:
        def loader(location, backend):
            if backend == 'constant':
                from zshar.models.constant_backend import ConstantAttributeBackend
                return ConstantAttributeBackend.load(str(location))
            from autogluon.tabular import TabularPredictor
            return TabularPredictor.load(str(location), verbosity=0)
    probability = np.zeros((len(frame), len(attributes), 3), dtype=np.float64)
    orders, calls = [], []
    for j, attr in enumerate(attributes):
        entry = manifest['models'][attr]
        location = model_dir / entry['path']
        started = time.monotonic()
        predictor = loader(location, entry['backend'])
        if entry['backend'] != 'constant':
            metadata = previous.read_json(location / 'backend_metadata.json')
            if metadata['feature_names'] != feature_columns:
                raise ValueError('Saved feature schema differs from the test input')
            if predictor.model_best != entry['metadata']['predictor_model_best']:
                raise ValueError('Saved best model has changed')
        native = predictor.predict_proba(frame[feature_columns])
        if not native.index.equals(frame.index) or native.columns.duplicated().any():
            raise ValueError('Misaligned inference rows/classes')
        order = [LEVELS.index(str(c)) for c in native.columns]
        probability[:, j, order] = native.to_numpy(dtype=float)
        orders.append(order)
        calls.append({'attribute':attr, 'backend':entry['backend'], 'model_path':str(location),
                      'load_calls':1, 'predict_proba_calls':1, 'test_samples':len(frame),
                      'native_class_order':[LEVELS[k] for k in order],
                      'probability_sha256':hashlib.sha256(probability[:,j,:].tobytes()).hexdigest(),
                      'seconds':time.monotonic()-started})
        del native, predictor
    return probability, orders, calls


def save_array(path, **arrays):
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def evaluate_configuration(directory, variant, selected, signature, model_dir, manifest,
                           frame, features, prototypes, candidates, weights, old_probability,
                           old_prediction, all_attributes, resume):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/'metrics.json'
    variant_signature = {**signature, 'variant':variant, 'selected_attributes':selected,
                         'protocol':'independent_model_reload_and_fresh_predict_proba_per_configuration'}
    if resume and result_path.exists():
        saved = previous.read_json(result_path)
        if saved['signature'] != variant_signature:
            raise ValueError('Completed configuration inputs changed')
        if previous.sha256(directory/'fresh_probabilities.npz') != saved['probability_npz_sha256']:
            raise ValueError('Completed inference output changed')
        with np.load(directory/'activity_prediction.npz') as data:
            prediction = data['prediction']
        return saved, prediction
    if result_path.exists():
        raise FileExistsError(result_path)
    # Crucially, this call runs separately for full9 and EACH removal variant.
    probability, orders, calls = fresh_attribute_probabilities(model_dir, manifest, frame, features, selected)
    selected_indices = [all_attributes.index(a) for a in selected]
    selected_prototypes = prototypes[:, selected_indices]
    selected_weights = [weights[a] for a in selected]
    components = ordinal_distance_components(probability, selected_prototypes, orders)
    scores = ordinal_scores(components, selected_weights)
    prediction = candidate_predictions(scores, candidates)
    metric = classification_summary(frame.activity_name, prediction)
    compared_probability = old_probability[:,selected_indices,:]
    difference = np.abs(probability-compared_probability)
    comparison = {'dataset':signature['dataset'], 'split_index':signature['split_index'],
        'variant':variant, 'selected_attribute_count':len(selected),
        'fresh_model_load_calls':len(calls), 'fresh_predict_proba_calls':len(calls),
        'autogluon_load_calls':sum(c['backend']=='autogluon' for c in calls),
        'constant_backend_load_calls':sum(c['backend']=='constant' for c in calls),
        'probabilities_exactly_equal_to_previous':bool(np.array_equal(probability,compared_probability)),
        'probability_max_abs_difference':float(difference.max()),
        'probability_values_changed':int(np.count_nonzero(difference)),
        'activity_predictions_changed':int(np.sum(prediction != old_prediction)),
        'top1_accuracy':metric['accuracy'], 'macro_f1':metric['macro_f1']}
    save_array(directory/'fresh_probabilities.npz',probabilities=probability)
    save_array(directory/'activity_prediction.npz',prediction=prediction)
    exported = frame[['window_index','activity_name']+selected].copy()
    for j,attr in enumerate(selected):
        for k,level in enumerate(LEVELS):
            exported[f'{attr}__p_{level}'] = probability[:,j,k]
    exported['predicted_activity'] = prediction
    exported.to_csv(directory/'fresh_test_predictions.csv.gz',index=False,compression='gzip')
    previous.write_json(directory/'inference_calls.json',calls)
    denominator = sum(selected_weights)
    saved = {'signature':variant_signature, 'top1_accuracy':metric['accuracy'],
        'macro_f1':metric['macro_f1'], 'comparison':comparison,
        'native_class_orders':orders, 'candidate_order':candidates,
        'probability_npz_sha256':previous.sha256(directory/'fresh_probabilities.npz'),
        'effective_weights':{a:weights[a]/denominator for a in selected}}
    previous.write_json(result_path,saved)
    del probability, components, scores
    gc.collect()
    return saved, prediction


def run_dataset(ds, output_string, resume, threads):
    import torch
    torch.set_num_threads(threads)
    output = Path(output_string)
    spec, run, splits, samples, _, _ = previous.source_inputs(ds)
    table = load_dataset_table(spec)
    attributes = previous.attribute_names()
    if table.attribute_columns != attributes:
        raise ValueError('Unexpected initial9 schema')
    all_rows, all_comparisons = [], []
    for split in splits:
        started = time.monotonic()
        directory = output/ds/f'split_{split.index:03d}'
        directory.mkdir(parents=True,exist_ok=True)
        frame = sample_evaluation_rows(table.frame,split,scenario='zsl',
            samples_per_activity=spec.eval_samples_per_activity,seed=42,manifest_ids=samples[split.index])
        if set(frame.window_index) != samples[split.index]:
            raise ValueError('Test windows differ from the original manifest')
        model_dir = run/f'split_{split.index:03d}/models'
        manifest = previous.read_json(model_dir/'attribute_backend_manifest.json')
        weights = validation_attribute_weights(attributes,manifest['validation_report'])
        candidates = list(split.unseen_activities)
        prototypes = np.array([[LEVELS.index(table.activity_vectors[c][a]) for a in attributes] for c in candidates])
        signature = {'dataset':ds,'split_index':int(split.index),
            'model_manifest_sha256':previous.sha256(model_dir/'attribute_backend_manifest.json'),
            'vectors_sha256':previous.sha256(spec.attributes_path),
            'test_window_indices':list(map(int,frame.window_index)),
            'test_features_sha256':hashlib.sha256(np.ascontiguousarray(frame[table.feature_columns].to_numpy()).tobytes()).hexdigest()}
        old_directory = OLD/ds/f'split_{split.index:03d}'
        old_signature = previous.read_json(old_directory/'metrics.json')['signature']
        if signature != old_signature:
            raise ValueError('New and previous experiments do not have the same inputs')
        # Previous outputs are ONLY used for comparison, never supplied to matching.
        with np.load(old_directory/'test_probabilities.npz') as data:
            old_probability = data['probabilities']
        old_activity = pd.read_csv(old_directory/'activity_predictions.csv.gz').set_index('window_index').loc[frame.window_index]
        old_rows = {r['variant']:r for r in previous.read_json(old_directory/'metrics.json')['rows']}
        predictions = frame[['window_index','user_id','activity_id','activity_name']].copy()
        rows, comparisons = [], []
        baseline = None
        for variant in ['full9']+attributes:
            selected = attributes if variant=='full9' else [a for a in attributes if a != variant]
            expected = 9 if variant=='full9' else 8
            if len(selected) != expected:
                raise ValueError('Incorrect retained attribute count')
            saved,prediction = evaluate_configuration(directory/variant,variant,selected,signature,
                model_dir,manifest,frame,table.feature_columns,prototypes,candidates,weights,
                old_probability,old_activity[variant+'__prediction'].to_numpy(dtype=str),attributes,resume)
            if variant=='full9':baseline=saved
            comparison=saved['comparison'].copy()
            comparison['top1_change_from_previous_pp']=100*(saved['top1_accuracy']-old_rows[variant]['top1_accuracy'])
            comparison['macro_f1_change_from_previous_pp']=100*(saved['macro_f1']-old_rows[variant]['macro_f1'])
            comparisons.append(comparison)
            predictions[variant+'__prediction']=prediction
            predictions[variant+'__correct']=prediction==frame.activity_name.to_numpy()
            if variant=='full9':predictions['full9_prediction']=prediction
            rows.append({'dataset':ds,'split_index':int(split.index),'variant':variant,
                'removed_attribute':'' if variant=='full9' else variant,'attribute_count':expected,
                'samples':len(frame),'candidate_count':len(candidates),
                'top1_accuracy':saved['top1_accuracy'],'macro_f1':saved['macro_f1'],
                'baseline_top1_accuracy':baseline['top1_accuracy'],'baseline_macro_f1':baseline['macro_f1'],
                'delta_top1':baseline['top1_accuracy']-saved['top1_accuracy'],
                'delta_macro_f1':baseline['macro_f1']-saved['macro_f1']})
        predictions.to_csv(directory/'activity_predictions.csv.gz',index=False,compression='gzip')
        previous.write_json(directory/'metrics.json',{'signature':signature,'rows':rows})
        pd.DataFrame(comparisons).to_csv(directory/'new_vs_previous.csv',index=False)
        all_rows.extend(rows);all_comparisons.extend(comparisons)
        print(f'DONE {ds} split={split.index} configurations=10 fresh_loads=81 '
              f'changed_predictions={sum(c["activity_predictions_changed"] for c in comparisons)} '
              f'seconds={time.monotonic()-started:.1f}',flush=True)
    return all_rows,all_comparisons


def summarize_new(output, rows, comparisons):
    dependency = CODE_ROOT/'new_results/initial9_seen_predictability/cache/python_deps'
    if dependency.exists():sys.path.insert(0,str(dependency))
    previous.summarize(output,rows,previous.DATASETS)
    report=output/'REPORT_zh.md'
    text=report.read_text().replace('在同一个 split、同一批窗口、同一候选类别和固定的概率输出上作配对比较',
        '在同一个 split、同一批窗口、同一候选类别上，各配置独立重新加载保留属性分类器并重新预测概率，作配对比较')
    text=text.replace('# 初始9维属性的逐属性移除实验','# 初始9维属性的逐属性移除实验：每次独立重载分类器并重新预测')
    text+='\n\n## 本次重新预测协议\n\n完整9维独立加载9个属性分类器；九个删除配置各独立加载8个保留属性分类器并调用 predict_proba。配置之间不复用分类器对象或概率输出。常量属性使用重新加载的常量预测器。旧概率仅用于事后比较，不参与本次匹配。没有重新训练或选模。\n\n各配置的 `inference_calls.json` 保存实际模型加载和预测调用记录；`fresh_probabilities.npz` 与 `fresh_test_predictions.csv.gz` 保存本次输出。\n'
    report.write_text(text)
    compared=pd.DataFrame(comparisons)
    compared.to_csv(output/'new_vs_previous.csv',index=False)
    for metric in ['top1','macro_f1']:
        new=pd.read_csv(output/f'{metric}_delta_mean_pp_9x5.csv',index_col=0)
        old=pd.read_csv(OLD/f'{metric}_delta_mean_pp_9x5.csv',index_col=0)
        (new-old).to_csv(output/f'{metric}_delta_difference_from_previous_9x5.csv')
    check={'complete':len(rows)==1000,'configurations':len(rows),
        'fresh_model_load_calls':int(compared.fresh_model_load_calls.sum()),
        'fresh_predict_proba_calls':int(compared.fresh_predict_proba_calls.sum()),
        'autogluon_load_calls':int(compared.autogluon_load_calls.sum()),
        'constant_backend_load_calls':int(compared.constant_backend_load_calls.sum()),
        'configurations_with_identical_probabilities':int(compared.probabilities_exactly_equal_to_previous.sum()),
        'maximum_probability_abs_difference':float(compared.probability_max_abs_difference.max()),
        'activity_predictions_changed':int(compared.activity_predictions_changed.sum()),
        'maximum_top1_change_from_previous_pp':float(compared.top1_change_from_previous_pp.abs().max()),
        'maximum_macro_f1_change_from_previous_pp':float(compared.macro_f1_change_from_previous_pp.abs().max())}
    previous.write_json(output/'reinference_checks.json',check)
    with (output/'REPORT_zh.md').open('a') as f:
        f.write('\n## 与旧实验的实测比较\n\n'+json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(check,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    output=validate_system_output(args.output)
    inputs={ds:previous.source_inputs(ds) for ds in previous.DATASETS}
    previous.write_json(output/'protocol.json',{'experiment':'fresh_model_reload_and_reprediction_per_LOAO_configuration',
        'output_filesystem':'system','datasets':previous.DATASETS,'seed':42,'splits_per_dataset':20,
        'configuration_count':1000,'expected_fresh_load_and_predict_calls':8100,
        'full9_classifiers_per_configuration':9,'ablated_classifiers_per_configuration':8,
        'cross_configuration_probability_reuse':False,'cross_configuration_model_object_reuse':False,
        'training':False,'workers':args.workers,'threads_per_worker':args.threads,
        'old_outputs_used_only_for_posthoc_comparison':str(OLD),
        'sources':{ds:{'run':str(v[1]),'vectors_sha256':previous.sha256(v[0].attributes_path),
                      'sample_manifest_sha256':previous.sha256(v[4])} for ds,v in inputs.items()}})
    if args.check:print('Preflight passed: 5 datasets, 100 splits, 1000 configurations, 8100 fresh calls.');return
    rows=[];comparisons=[]
    if args.summarize_only:
        for ds in previous.DATASETS:
            for split in inputs[ds][2]:
                folder=output/ds/f'split_{split.index:03d}'
                rows.extend(previous.read_json(folder/'metrics.json')['rows'])
                comparisons.extend(pd.read_csv(folder/'new_vs_previous.csv').to_dict('records'))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run_dataset,ds,str(output),args.resume,args.threads) for ds in previous.DATASETS]
            for future in as_completed(futures):
                new_rows,new_comparisons=future.result();rows.extend(new_rows);comparisons.extend(new_comparisons)
    summarize_new(output,rows,comparisons)
    print('FINISHED '+str(output),flush=True)


if __name__=='__main__':main()
