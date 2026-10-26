#!/usr/bin/env python3
"""Overwrite current MHEALTH initial-prototype geometry and dependent comparisons."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from attribute_discrimination_coverage import read_vectors,compute_attribute_coverage
from prototype_distance_heatmap import encode_prototypes,pairwise_prototype_distances
from compare_refinement_prototypes import plot_matrices,METRICS,NAMES

CODE=Path(__file__).resolve().parent
PROJECT=Path('/home/wanzhen/myproject_new')
SOURCE=PROJECT/'configs/attributes_new/activity_attribute_vectors_initial_mhealth_v1_5.json'
ORIGINAL=PROJECT/'configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json'


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def matrix(path):return pd.read_csv(path,index_col=0,float_precision='round_trip')


def updated_distances(old_vectors,new_vectors):
    old=encode_prototypes(old_vectors);new=encode_prototypes(new_vectors)
    if old.activities!=new.activities or old.attributes!=new.attributes or len(new.attributes)!=9:
        raise ValueError('Activity/attribute order changed')
    j=new.attributes.index('motion_regularity')
    keep=[k for k in range(9) if k!=j]
    np.testing.assert_array_equal(old.values[:,keep],new.values[:,keep])
    if not np.all(old.values[:,j]==1):raise ValueError('Original regularity must be uniformly High')
    mask=new.values[:,j,None]!=new.values[None,:,j]
    old_distances={m:pairwise_prototype_distances(old,metric=m) for m in METRICS}
    distances={m:pairwise_prototype_distances(new,metric=m) for m in METRICS}
    np.testing.assert_allclose(distances['ordinal_probability']-old_distances['ordinal_probability'],mask/18,atol=1e-14,rtol=0)
    np.testing.assert_allclose(distances['euclidean']**2-old_distances['euclidean']**2,mask/36,atol=1e-14,rtol=0)
    for metric,frame in distances.items():
        delta=np.abs(new.values[:,None,:]-new.values[None,:,:])
        expected=delta.mean(2) if metric=='ordinal_probability' else np.sqrt((delta**2).mean(2))
        np.testing.assert_allclose(frame,expected,atol=1e-14,rtol=0)
        np.testing.assert_allclose(frame,frame.T,atol=1e-14,rtol=0)
        assert np.all(np.diag(frame)==0) and np.all((frame.to_numpy()>=0)&(frame.to_numpy()<=1))
    return new,old_distances,distances,int(np.triu(mask,1).sum())


def main(args):
    root=args.root.resolve()
    if not root.is_relative_to((CODE/'new_results').resolve()) or root.stat().st_dev!=Path('/').stat().st_dev:
        raise ValueError('Use the existing current analysis on the actual system filesystem')
    if root.stat().st_dev==Path('/data').stat().st_dev:raise ValueError('System filesystem required')
    cache=root/'_cache/mhealth_initial_revision';cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache/'matplotlib');os.environ['TMPDIR']=str(cache)
    before_dir=root/'before/mhealth';after_dir=root/'after/mhealth';comparison=root/'comparison/mhealth'
    for directory in [before_dir,after_dir,comparison]:
        if not directory.is_dir():raise FileNotFoundError(directory)
    vectors=read_vectors(SOURCE);old_vectors=read_vectors(ORIGINAL)
    table,old_distances,distances,changed_pairs=updated_distances(old_vectors,vectors)
    if changed_pairs!=32:raise ValueError('Expected 32 changed unordered activity pairs')
    # Never rerun A2 or alter other dataset outputs. Hash only the relevant current geometry trees.
    protected={}
    for stage in ['before','after','comparison']:
        for ds in ['pamap2','uschad','wisdm','dsads']:
            for p in (root/stage/ds).rglob('*'):
                if p.is_file():protected[str(p)]=sha(p)
    for p in after_dir.rglob('*'):
        if p.is_file() and p.name not in ['distance_change_matrix.csv','euclidean_distance_change_matrix.csv']:
            protected[str(p)]=sha(p)
    source_hashes={str(p.resolve()):sha(p) for p in [SOURCE,ORIGINAL]}
    previous_entry=next(e for e in read_json(root/'run_metadata.json')['inputs'] if e.get('dataset')=='mhealth' and e.get('stage')=='before')
    # Detect unrelated baseline edits before overwriting any artifact.
    existing=matrix(before_dir/'distance_matrix.csv').loc[list(table.activities),list(table.activities)]
    if not any(np.allclose(existing,d,atol=1e-14,rtol=0) for d in [old_distances['ordinal_probability'],distances['ordinal_probability']]):
        raise ValueError('Current baseline is neither the original nor this reviewed initial version')
    save_json(before_dir/'vectors_snapshot.json',vectors)
    pd.DataFrame(table.values,index=table.activities,columns=table.attributes).to_csv(before_dir/'normalized_prototypes.csv',index_label='activity')
    coverage=pd.DataFrame(compute_attribute_coverage(vectors)).set_index('attribute')
    coverage.to_csv(before_dir/'attribute_coverage.csv')
    summary=pd.read_csv(root/'dataset_summary.csv',float_precision='round_trip')
    target=summary.dataset=='mhealth'
    summary.loc[target,'before_mean_attribute_coverage']=coverage.coverage.mean()
    triangles=np.triu_indices(len(table.activities),1)
    for metric,frame in distances.items():
        eu=metric=='euclidean';prefix='euclidean_' if eu else ''
        filename=prefix+'distance_matrix.csv'
        frame.to_csv(before_dir/filename,index_label='activity')
        stem='euclidean_distance_heatmap' if eu else 'prototype_distance_heatmap'
        plot_matrices([frame],['MHEALTH: revised initial 9D (regularity review)'],before_dir/stem,METRICS[metric],args.dpi)
        refined=matrix(after_dir/filename).loc[list(table.activities),list(table.activities)]
        (refined-frame).to_csv(after_dir/(prefix+'distance_change_matrix.csv'),index_label='activity')
        plot_matrices([frame,refined],['MHEALTH: revised initial 9D','MHEALTH: existing A2 mean (20 splits; original A1 lineage)'],
                      comparison/(prefix+'prototype_distance_before_after'),METRICS[metric],args.dpi)
        summary.loc[target,metric+'_before_mean_distance']=float(frame.to_numpy()[triangles].mean())
    summary.to_csv(root/'dataset_summary.csv',index=False)
    # Synchronize baseline-dependent coverage outputs so snapshots do not disagree with geometry.
    before_cov=matrix(root/'before/attribute_coverage_matrix.csv')
    old_cov=before_cov.copy()
    before_cov['MHEALTH']=coverage.loc[before_cov.index,'coverage']
    before_cov.to_csv(root/'before/attribute_coverage_matrix.csv',index_label='attribute')
    plot_matrices([before_cov],['Initial 9D (MHEALTH regularity reviewed)'],root/'before/attribute_coverage_heatmap',
                  'Attribute discrimination coverage',args.dpi,coverage=True)
    after_cov=matrix(root/'after/attribute_coverage_matrix.csv').reindex(before_cov.index)
    counts=matrix(root/'after/attribute_presence_counts.csv').reindex(before_cov.index)
    (after_cov-before_cov).to_csv(root/'comparison/shared_attributes_coverage_change.csv',index_label='attribute')
    plot_matrices([before_cov,after_cov],['Initial 9D (MHEALTH revised)','Existing A2 conditional mean (original A1 lineage)'],
        root/'comparison/shared_attributes_coverage_before_after','Attribute discrimination coverage',args.dpi,coverage=True,
        counts=[pd.DataFrame(1,index=before_cov.index,columns=before_cov.columns),counts])
    records=pd.read_csv(root/'all_attribute_coverage.csv',float_precision='round_trip')
    for attr,row in coverage.iterrows():
        selected=(records.dataset=='mhealth')&(records.stage=='before')&(records.attribute==attr)
        if selected.sum()!=1:raise ValueError('Missing/duplicate initial coverage row')
        for field,value in row.items():records.loc[selected,field]=value
    records.to_csv(root/'all_attribute_coverage.csv',index=False)
    metadata=read_json(root/'run_metadata.json')
    for entry in metadata['inputs']:
        if entry.get('dataset')=='mhealth' and entry.get('stage')=='before':
            entry.update(source_path=str(SOURCE.resolve()),sha256=sha(SOURCE),
                revision='initial9_mhealth_motion_regularity_semantic_review_v1',
                historical_refinement_a1_sha256=sha(ORIGINAL))
    metadata['mhealth_initial_revision']={'updated_utc':datetime.now(timezone.utc).isoformat(),
        'source_path':str(SOURCE.resolve()),'sha256':sha(SOURCE),'update_script':str(Path(__file__).resolve()),
        'update_script_sha256':sha(Path(__file__)),'a2_refinement_lineage':'A2 remains generated from original A1; not regenerated using the reviewed initial prototypes',
        'scope':'MHEALTH initial geometry and baseline-dependent comparisons/coverage/summary only'}
    save_json(root/'run_metadata.json',metadata)
    readme=root/'README.md';text=readme.read_text(encoding='utf-8')
    start='<!-- MHEALTH_INITIAL_REVISION_START -->';end='<!-- MHEALTH_INITIAL_REVISION_END -->'
    if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
    note=f'''{start}
## MHEALTH 初始原型已更新

MHEALTH 初始9维输入现为 `configs/attributes_new/activity_attribute_vectors_initial_mhealth_v1_5.json`。
已覆盖 `before/mhealth/` 中的 L1 / RMS 热力图、距离矩阵和向量快照，并同步更新 MHEALTH 对照图及依赖初始基线的汇总。
其他数据集和 A2 原型、A2 距离均值/标准差保持原值。A2 的生成依据仍是旧版 A1，因此当前对照图展示“新版初始原型”与“既有 A2”的几何比较，不能描述为从新版初始向量重新细化得到的结果。
更新核验见 `before/mhealth/revision_verification.json`；重现当前覆盖操作：
`python -B /home/wanzhen/attribute_prototype_analysis/update_mhealth_initial_distance.py`。
下面原始批处理命令复现的是历史 A1/A2 分析，需再执行本更新脚本才能得到此处的新版 MHEALTH 初始结果。
{end}

'''
    readme.write_text(note+text.lstrip(),encoding='utf-8')
    assert all(sha(Path(p))==h for p,h in protected.items())
    assert all(sha(Path(p))==h for p,h in source_hashes.items())
    np.testing.assert_allclose(matrix(before_dir/'distance_matrix.csv'),distances['ordinal_probability'],atol=1e-14,rtol=0)
    # One coverage cell changes on first update; repeated application is idempotent.
    changed_cov=int((np.abs(before_cov-old_cov)>1e-14).to_numpy().sum())
    if changed_cov not in [0,1]:raise ValueError('Unexpected baseline coverage changes')
    verification={'status':'passed','source_path':str(SOURCE.resolve()),'input_sha256':sha(SOURCE),
        'activity_count':12,'attribute_count':9,'total_unordered_pairs':66,'changed_unordered_pairs':changed_pairs,
        'ordinal_l1_cross_group_increase':1/18,'euclidean_squared_cross_group_increase':1/36,
        'all_other_activity_pairs_unchanged':True,'symmetric_zero_diagonal_unit_range':True,
        'independent_pair_formula_checked':True,'other_dataset_and_a2_artifacts_unchanged':True,
        'protected_artifact_hash_count':len(protected),'source_files_unchanged':True,
        'previous_initial_entry':previous_entry,
        'old_mean_l1':float(old_distances['ordinal_probability'].to_numpy()[triangles].mean()),
        'new_mean_l1':float(distances['ordinal_probability'].to_numpy()[triangles].mean()),
        'old_mean_rms':float(old_distances['euclidean'].to_numpy()[triangles].mean()),
        'new_mean_rms':float(distances['euclidean'].to_numpy()[triangles].mean()),
        'a2_refinement_lineage_note':metadata['mhealth_initial_revision']['a2_refinement_lineage']}
    save_json(before_dir/'revision_verification.json',verification)
    root_check=read_json(root/'verification_report.json')
    root_check['mhealth_initial_revision']={'status':'passed','details':'before/mhealth/revision_verification.json',
        'source_sha256':sha(SOURCE),'historical_a2_generated_from_original_initial':True}
    save_json(root/'verification_report.json',root_check)
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print('Overwritten:',before_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=CODE/'new_results')
    parser.add_argument('--dpi',type=int,default=300)
    main(parser.parse_args())
