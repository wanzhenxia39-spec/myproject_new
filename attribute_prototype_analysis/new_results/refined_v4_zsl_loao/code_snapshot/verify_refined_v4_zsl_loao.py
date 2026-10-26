#!/usr/bin/env python3
"""Recompute predictions and both metrics from all persisted LOAO outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import reinfer_refined_v4_zsl_loao as exp


def main():
    output=exp.OUTPUT
    audit=exp.common.read_json(output/'input_audit.json')
    config_count=load_count=historical_count=identical_count=0
    semantic_groups={}
    for ds in exp.DATASETS:
        stable=[]
        semantic_set=set()
        for split,meta in audit[ds]['vectors'].items():
            attrs=meta['attributes']
            stable.append(set(attrs))
            semantic_set.update((a,exp.semantic_hash(v)) for a,v in meta['semantic_definitions'].items())
            directory=output/ds/f'split_{int(split):03d}'
            provenance=exp.common.read_json(directory/'input_provenance.json')
            vectors,_=exp.load_activity_vectors(meta['path'])
            with np.load(directory/'full/fresh_probabilities.npz') as payload:
                full_q=payload['probabilities']
            for variant in ['full']+attrs:
                target=directory/variant
                saved=exp.common.read_json(target/'metrics.json')
                selected=saved['signature']['selected_attributes']
                expected=attrs if variant=='full' else [a for a in attrs if a!=variant]
                assert selected==expected
                calls=exp.common.read_json(target/'inference_calls.json')
                assert [c['attribute'] for c in calls]==selected
                assert all(c['load_calls']==c['predict_proba_calls']==1 for c in calls)
                assert len(calls)==saved['fresh_load_calls']==saved['fresh_predict_proba_calls']
                for filename,h in saved['output_hashes'].items():
                    assert exp.common.digest(target/filename)==h
                with np.load(target/'fresh_probabilities.npz') as payload:
                    q=payload['probabilities']
                assert q.shape==(len(saved['signature']['test_window_indices']),len(selected),3)
                export=pd.read_csv(target/'fresh_test_predictions.csv.gz')
                assert export.window_index.tolist()==saved['signature']['test_window_indices']
                candidates=saved['candidate_order']
                proto=np.array([[exp.LEVELS.index(vectors[c][a]) for a in selected] for c in candidates])
                comp=exp.ordinal_distance_components(q,proto,saved['native_class_orders'])
                raw_weights=[provenance['weights'][a] for a in selected]
                prediction=exp.candidate_predictions(exp.ordinal_scores(comp,raw_weights),candidates)
                assert set(export.activity_name)==set(candidates)
                assert np.array_equal(prediction,export.predicted_activity.to_numpy(dtype=str))
                metrics=exp.classification_summary(export.activity_name,prediction)
                assert abs(metrics['accuracy']-saved['top1_accuracy'])<1e-12
                assert abs(metrics['macro_f1']-saved['macro_f1'])<1e-12
                assert abs(sum(saved['effective_weights'].values())-1)<1e-12
                assert set(saved['effective_weights'])==set(selected)
                for j,a in enumerate(selected):
                    assert export[a+'__target'].tolist()==[vectors[c][a] for c in export.activity_name]
                    for k,level in enumerate(exp.LEVELS):
                        assert np.allclose(export[a+'__p_'+level],q[:,j,k],atol=1e-15,rtol=0)
                if variant=='full':
                    old=pd.read_csv(Path(audit[ds]['run'])/f'split_{int(split):03d}/details.csv').set_index('window_index').loc[export.window_index]
                    assert np.array_equal(prediction,old.predicted_activity.to_numpy(dtype=str))
                    historical_count+=1
                else:
                    indices=[attrs.index(a) for a in selected]
                    identical_count+=int(np.array_equal(q,full_q[:,indices,:]))
                config_count+=1
                load_count+=len(calls)
        semantic_groups[ds]={'common_attributes_20_splits':exp.ordered(set.intersection(*stable)),
                             'unique_names':len(set.union(*stable)), 'exact_semantic_groups':len(semantic_set)}
    assert config_count==1105 and load_count==10145 and historical_count==100
    common=exp.common.read_json(output/'verification.json')
    assert common['configurations']==config_count
    assert common['fresh_model_load_calls']==load_count
    check={'passed':True,'recomputed_configurations':config_count,'audited_fresh_model_calls':load_count,
        'historical_full_prediction_matches':historical_count,'drops_with_exactly_equal_probabilities_to_full':identical_count,
        'schema_coverage':semantic_groups}
    exp.common.write_json(output/'artifact_verification.json',check)
    print(exp.canonical(check),flush=True)


if __name__=='__main__':main()
