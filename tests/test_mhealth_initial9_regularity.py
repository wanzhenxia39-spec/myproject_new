from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
import pytest
import run_mhealth_initial9_regularity as driver


def attrs():
    return [a['name'] for a in driver.read(driver.PROJECT/'configs/attributes/initial_attributes.json')['attributes']]


def synthetic_rows(dataset):
    rows=[]
    for split in range(20):
        for i,variant in enumerate(['full9']+attrs()):
            accuracy=.6-i*.003
            f1=.5-i*.004
            rows.append({'dataset':dataset,'split_index':split,'variant':variant,
                'removed_attribute':'' if variant=='full9' else variant,'attribute_count':9 if variant=='full9' else 8,
                'samples':120,'candidate_count':4,'top1_accuracy':accuracy,'macro_f1':f1,
                'baseline_top1_accuracy':.6,'baseline_macro_f1':.5,
                'delta_top1':.6-accuracy,'delta_macro_f1':.5-f1})
    return rows


def test_training_configuration_matches_historical_hyperparameters(tmp_path):
    reference=driver.read(driver.REFERENCE/'config.json')
    current=asdict(driver.build_config(tmp_path,reference,Path(reference['split_manifest']),Path(reference['sample_manifest']),reference['split_indices'],False))
    for key in reference:
        if key not in {'run_root','run_name','reuse_models','split_indices'}:
            assert current[key]==reference[key]
    assert current['backend_config']['model_suite']=='default'
    assert current['backend_config']['fit_weighted_ensemble'] is True
    assert set(current['split_indices'])==set(reference['split_indices'])
    assert current['run_name']!=reference['run_name']


def test_only_regularity_may_be_changed():
    old=driver.read(driver.PROJECT/'configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json')
    new=driver.read(driver.VECTORS)
    driver.validate_reannotation(old,new,attrs())
    invalid=deepcopy(new)
    invalid['walking']['motion_intensity']='Low'
    with pytest.raises(ValueError):driver.validate_reannotation(old,invalid,attrs())
    with pytest.raises(ValueError):driver.validate_reannotation(old,old,attrs())


def test_resume_accepts_json_set_order_but_rejects_hyperparameter_changes():
    old={'seed':42,'split_indices':[53,13,17],'reuse_models':False,'backend_config':{'time_limit':300}}
    new={'seed':42,'split_indices':{13,17,53},'reuse_models':True,'backend_config':{'time_limit':300}}
    driver.assert_compatible_config(old,new)
    new['backend_config']['time_limit']=301
    with pytest.raises(ValueError):driver.assert_compatible_config(old,new)


def test_wisdm_and_unpaired_deltas_are_rejected():
    rows=[r for ds in driver.DATASETS for r in synthetic_rows(ds)]
    driver.validate_rows(rows,driver.DATASETS)
    with pytest.raises(ValueError):driver.validate_rows(rows+synthetic_rows('wisdm'),driver.DATASETS)
    rows[1]['delta_top1']=.5
    with pytest.raises(ValueError):driver.validate_rows(rows,driver.DATASETS)


def test_old_constant_regularity_model_is_rejected(tmp_path):
    from zshar.pipeline.training import split_train_validation
    names=attrs()
    frame=pd.DataFrame({'activity_name':['steady']*20+['discrete']*20})
    for attr in names:frame[attr]='Low'
    frame[driver.ATTR]=['High']*20+['Medium']*20
    fit,_=split_train_validation(frame,seed=42,validation_fraction=.3)
    counts={a:{str(k):int(v) for k,v in fit[a].value_counts().items()} for a in names}
    manifest={'manager_type':'independent','classifier_backend':'autogluon','attribute_cols':names,'class_counts':deepcopy(counts)}
    manifest['class_counts'][driver.ATTR]={'High':len(fit)}
    driver.write(tmp_path/'attribute_backend_manifest.json',manifest)
    table=SimpleNamespace(frame=frame,attribute_columns=names)
    split=SimpleNamespace(seen_activities=['steady','discrete'])
    with pytest.raises(ValueError):driver.verify_model_manifest(tmp_path,table,split)
    manifest['class_counts']=counts
    driver.write(tmp_path/'attribute_backend_manifest.json',manifest)
    assert driver.verify_model_manifest(tmp_path,table,split)['class_counts']==counts


def test_system_disk_output_is_rejected():
    with pytest.raises(ValueError):driver.data_output('/home/wanzhen/attribute_prototype_analysis/new_results/forbidden_model_output')


def test_four_dataset_plot_smoke_without_models(tmp_path,monkeypatch):
    historical=tmp_path/'historical'
    monkeypatch.setattr(driver,'HISTORICAL_LOAO',historical)
    root=tmp_path/'new'
    for ds in driver.DATASETS:
        directory=(root/'loao_4datasets' if ds=='mhealth' else historical)/ds
        all_rows=synthetic_rows(ds)
        for index in range(20):
            rows=[r for r in all_rows if r['split_index']==index]
            driver.write(directory/f'split_{index:03d}/metrics.json',
                {'signature':{'vectors_sha256':driver.sha(driver.VECTORS)},'rows':rows})
    (historical/'wisdm').mkdir()
    (historical/'wisdm/should_not_be_read.json').write_text('invalid JSON')
    driver.plot_four(root,{'test_fixture':True})
    output=root/'loao_4datasets'
    assert (output/'loao_macro_f1_impact_4datasets.png').exists()
    assert (output/'loao_top1_impact_4datasets.png').exists()
    assert not list(output.glob('*5datasets*'))
    matrix=pd.read_csv(output/'macro_f1_delta_mean_pp_9x4.csv',index_col=0)
    assert list(matrix.columns)==driver.DATASETS
    assert matrix.shape==(9,4)
    summary=pd.read_csv(output/'per_split_metrics.csv')
    assert len(summary)==800 and 'wisdm' not in set(summary.dataset)
    assert (output/'initial9_zsl_loao_results.xlsx').exists()
    focus=driver.read(output/'motion_regularity_effect.json')
    assert focus['attribute']=='motion_regularity' and focus['split_count']==20
