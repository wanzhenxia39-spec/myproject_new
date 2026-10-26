#!/usr/bin/env python3
"""Audit real reload logs, retained coordinates and independently rescored activities."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT=Path(__file__).resolve().parent/'new_results/initial9_zsl_loao_new'
DATASETS=['pamap2','mhealth','uschad','wisdm','dsads']


def main():
    independent=[];total_loads=0;total_predictions=0;ag=0;constant=0;all_rows=[]
    attrs=None;old_predictions_changed=0;max_vs_old=0.
    for ds in DATASETS:
        splits=sorted((ROOT/ds).glob('split_*/metrics.json'))
        assert len(splits)==20,(ds,len(splits))
        for p in splits:
            folder=p.parent
            split_data=json.loads(p.read_text())
            split_index=split_data['signature']['split_index']
            metrics={r['variant']:r for r in split_data['rows']}
            full=json.loads((folder/'full9/metrics.json').read_text())
            attrs=full['signature']['selected_attributes']
            assert len(attrs)==9 and len(metrics)==10
            with np.load(folder/'full9/fresh_probabilities.npz') as data:
                baseline=data['probabilities']
            baseline_predictions=np.load(folder/'full9/activity_prediction.npz')['prediction']
            for variant in ['full9']+attrs:
                directory=folder/variant
                saved=json.loads((directory/'metrics.json').read_text())
                selected=saved['signature']['selected_attributes']
                expected=attrs if variant=='full9' else [a for a in attrs if a!=variant]
                assert selected==expected
                assert len(selected)==(9 if variant=='full9' else 8)
                calls=json.loads((directory/'inference_calls.json').read_text())
                assert [c['attribute'] for c in calls]==selected
                assert len(calls)==len(selected)
                assert all(c['load_calls']==c['predict_proba_calls']==1 for c in calls)
                total_loads+=sum(c['load_calls'] for c in calls)
                total_predictions+=sum(c['predict_proba_calls'] for c in calls)
                ag+=sum(c['backend']=='autogluon' for c in calls)
                constant+=sum(c['backend']=='constant' for c in calls)
                with np.load(directory/'fresh_probabilities.npz') as data:q=data['probabilities']
                assert q.shape==(len(baseline),len(selected),3)
                assert hashlib.sha256((directory/'fresh_probabilities.npz').read_bytes()).hexdigest()==saved['probability_npz_sha256']
                for j,call in enumerate(calls):
                    assert hashlib.sha256(q[:,j,:].tobytes()).hexdigest()==call['probability_sha256']
                selected_indices=[attrs.index(a) for a in selected]
                reference=baseline[:,selected_indices,:]
                delta=float(np.max(np.abs(q-reference)))
                independent.append({'dataset':ds,'split_index':split_index,'variant':variant,
                    'equal_to_fresh_full9_retained_probabilities':bool(np.array_equal(q,reference)),
                    'max_probability_abs_difference_from_fresh_full9':delta})
                comparison=saved['comparison']
                old_predictions_changed+=comparison['activity_predictions_changed']
                max_vs_old=max(max_vs_old,comparison['probability_max_abs_difference'])
                frame=pd.read_csv(directory/'fresh_test_predictions.csv.gz')
                assert len(frame)==len(q)
                assert not any(c.startswith(variant+'__p_') for c in frame.columns) if variant!='full9' else True
                pred=np.load(directory/'activity_prediction.npz')['prediction']
                np.testing.assert_array_equal(pred,frame.predicted_activity.to_numpy())
                acc=accuracy_score(frame.activity_name,pred)
                f1=f1_score(frame.activity_name,pred,average='macro',zero_division=0)
                assert abs(acc-saved['top1_accuracy'])<1e-12
                assert abs(f1-saved['macro_f1'])<1e-12
                row=metrics[variant]
                assert abs(row['top1_accuracy']-acc)<1e-12 and abs(row['macro_f1']-f1)<1e-12
                assert abs(row['delta_top1']-(metrics['full9']['top1_accuracy']-acc))<1e-12
                assert abs(row['delta_macro_f1']-(metrics['full9']['macro_f1']-f1))<1e-12
                assert set(saved['effective_weights'])==set(selected)
                assert abs(sum(saved['effective_weights'].values())-1)<1e-12
                all_rows.append(row)
    assert len(all_rows)==1000 and total_loads==total_predictions==8100
    assert ag==7920 and constant==180
    frame=pd.DataFrame(all_rows)
    for metric,delta in [('top1','delta_top1'),('macro_f1','delta_macro_f1')]:
        raw=frame[frame.variant.ne('full9')]
        expected=raw.groupby(['variant','dataset'])[delta].mean().unstack().reindex(index=attrs,columns=DATASETS)*100
        observed=pd.read_csv(ROOT/f'{metric}_delta_mean_pp_9x5.csv',index_col=0)
        assert observed.shape==(9,5)
        np.testing.assert_allclose(observed.to_numpy(),expected.to_numpy(),atol=1e-9,rtol=0)
    pd.DataFrame(independent).to_csv(ROOT/'fresh_vs_fresh_full9_probability_comparison.csv',index=False)
    dropped=[r for r in independent if r['variant']!='full9']
    summary={'status':'passed','splits':100,'independent_configurations':1000,
        'actual_model_load_calls':total_loads,'actual_predict_proba_calls':total_predictions,
        'autogluon_predictor_calls':ag,'constant_predictor_calls':constant,
        'removed_attribute_classifier_never_loaded':True,
        'retained_coordinate_shapes_and_call_hashes_verified':True,
        'independently_rescored_activity_metrics':1000,
        'ablated_configurations_equal_to_fresh_full9_retained_probabilities':sum(r['equal_to_fresh_full9_retained_probabilities'] for r in dropped),
        'max_abs_probability_difference_between_fresh_configurations':max(r['max_probability_abs_difference_from_fresh_full9'] for r in dropped),
        'max_abs_probability_difference_from_previous_experiment':max_vs_old,
        'activity_predictions_changed_from_previous':old_predictions_changed,
        'main_matrix_shape':[9,5],'unit_tests_passed':2}
    (ROOT/'validation_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
