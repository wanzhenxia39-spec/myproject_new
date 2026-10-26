import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest
import reinfer_refined_v4_zsl_loao as experiment


@pytest.mark.parametrize('dimension',[9,10,11,12])
def test_varying_dimensions_and_all_removals_match_original(dimension):
    rng=np.random.default_rng(42+dimension)
    q=rng.dirichlet([1,2,3],size=(4,dimension))
    attributes=[f'a{i}' for i in range(dimension)]
    candidates=['activity_a','activity_b','activity_c']
    proto=rng.integers(0,3,size=(3,dimension))
    vectors={c:{a:experiment.LEVELS[proto[i,j]] for j,a in enumerate(attributes)} for i,c in enumerate(candidates)}
    weights={a:float(rng.uniform(.1,1)) for a in attributes}
    for removed in [None]+list(range(dimension)):
        indices=[i for i in range(dimension) if i!=removed]
        selected=[attributes[i] for i in indices]
        selected_q=q[:,indices,:]
        orders=[[0,1,2] for _ in indices]
        comp=experiment.ordinal_distance_components(selected_q,proto[:,indices],orders)
        scores=experiment.ordinal_scores(comp,[weights[a] for a in selected])
        experiment.verify_original_scores(selected_q,orders,proto[:,indices],selected,candidates,
                                          {a:weights[a] for a in selected},vectors,scores)


def test_missing_attribute_is_not_counted_as_zero_and_singleton_sd_is_nan():
    records=[]
    for split,attribute,delta in [(1,'common',2),(2,'common',-1),(1,'rare',-3)]:
        records.append({'dataset':'x','split_index':split,'removed_attribute':attribute,
            'delta_top1_pp':delta,'delta_macro_f1_pp':delta,'baseline_top1_accuracy':.6,
            'baseline_macro_f1':.5,'top1_accuracy':.6-delta/100,'macro_f1':.5-delta/100,
            'semantic_sha256':'stable'})
    g=experiment.aggregate_deltas(pd.DataFrame(records),['dataset','removed_attribute']).set_index('removed_attribute')
    assert g.loc['rare','n_splits']==1
    assert g.loc['rare','top1_delta_mean_pp']==-3
    assert np.isnan(g.loc['rare','top1_delta_sd_pp'])
    assert g.loc['common','top1_delta_mean_pp']==.5
    assert g.loc['common','top1_delta_sd_pp']==pytest.approx(np.std([2,-1],ddof=1))


def test_each_configuration_reloads_only_selected_models():
    calls=[]
    class Fake:
        def predict_proba(self,frame):
            calls.append(('predict',len(frame)))
            return pd.DataFrame({'Low':np.ones(len(frame))},index=frame.index)
    def loader(path,backend):
        calls.append(('load',Path(path).name))
        return Fake()
    manifest={'models':{a:{'path':a,'backend':'constant'} for a in ['a','b','c']}}
    frame=pd.DataFrame({'imu':[1.,2.]})
    for selected in [['a','b','c'],['b','c'],['a','c'],['a','b']]:
        start=len(calls)
        q,orders,audit=experiment.fresh_attribute_probabilities(Path('/fake'),manifest,frame,['imu'],selected,loader)
        assert calls[start:][::2]==[('load',a) for a in selected]
        assert q.shape==(2,len(selected),3)
        assert len(audit)==len(selected)
    assert sum(c[0]=='load' for c in calls)==9


def test_same_name_different_definitions_remain_separate_in_exact_summary():
    rows=[]
    for split,sem,delta in [(1,'definition_a',2),(2,'definition_b',-2)]:
        rows.append({'dataset':'x','split_index':split,'removed_attribute':'same_name',
            'semantic_sha256':sem,'delta_top1_pp':delta,'delta_macro_f1_pp':delta,
            'baseline_top1_accuracy':.5,'baseline_macro_f1':.5,
            'top1_accuracy':.5-delta/100,'macro_f1':.5-delta/100})
    frame=pd.DataFrame(rows)
    pooled=experiment.aggregate_deltas(frame,['dataset','removed_attribute'])
    exact=experiment.aggregate_deltas(frame,['dataset','removed_attribute','semantic_sha256'])
    assert pooled.semantic_variants.iloc[0]==2
    assert pooled.top1_delta_mean_pp.iloc[0]==0
    assert len(exact)==2
    assert set(exact.top1_delta_mean_pp)=={-2,2}
    assert exact.n_splits.eq(1).all()
