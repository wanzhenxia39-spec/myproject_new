from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from analyze_initial9_loao_by_activity import class_metrics,paired_summary


def test_class_f1_keeps_false_positives_from_other_true_activities():
    truth=np.array(['a','a','b','b','c','c'])
    pred=np.array(['a','b','a','b','a','c'])
    rows={r['activity']:r for r in class_metrics(truth,pred,['a','b','c'])}
    a=rows['a']
    assert (a['tp'],a['fn'],a['fp'],a['tn'])==(1,1,2,2)
    assert a['top1_accuracy']==.5
    assert a['class_f1']==pytest.approx(.4)
    assert a['ovr_macro_f1']==pytest.approx(f1_score(truth=='a',pred=='a',average='macro'))
    assert np.mean([r['class_f1'] for r in rows.values()])==pytest.approx(f1_score(truth,pred,average='macro'))


def test_never_recognized_class_gets_zero_accuracy_and_f1():
    rows={r['activity']:r for r in class_metrics(['a','a','b','b'],['b']*4,['a','b'])}
    assert rows['a']['top1_accuracy']==0
    assert rows['a']['class_f1']==0
    assert rows['a']['fn']==2


def test_unseen_candidate_set_is_enforced():
    with pytest.raises(ValueError):class_metrics(['a','b'],['seen','a'],['a','b'])


def test_condition_on_unseen_occurrence_and_pair_with_same_split():
    rows=[]
    for split,activity,full,drop in [(1,'a',.8,.6),(2,'b',.5,.7),(3,'a',.4,.5)]:
        for variant,value in [('full9',full),('attribute_x',drop)]:
            rows.append({'split_index':split,'activity':activity,'variant':variant,'support':50,
                         'top1_accuracy':value,'class_f1':value,'ovr_macro_f1':value})
    paired,summary=paired_summary(pd.DataFrame(rows),['attribute_x'])
    g=summary.set_index('activity')
    assert g.loc['a','n_unseen_splits']==2
    assert g.loc['a','unseen_split_indices']=='1,3'
    assert g.loc['a','delta_top1_accuracy_mean_pp']==pytest.approx(5.)
    assert g.loc['a','delta_top1_accuracy_sd_pp']==pytest.approx(np.std([20,-10],ddof=1))
    assert g.loc['a','full_top1_accuracy_mean_pct']==pytest.approx(60.)
    assert g.loc['a','removed_top1_accuracy_mean_pct']==pytest.approx(55.)
    assert g.loc['a','removed_minus_full_top1_accuracy_mean_pp']==pytest.approx(-5.)
    assert g.loc['b','delta_top1_accuracy_mean_pp']==pytest.approx(-20.)
    assert g.loc['b','n_unseen_splits']==1
    assert np.isnan(g.loc['b','delta_top1_accuracy_sd_pp'])
    assert len(paired)==3


def test_missing_full_baseline_is_rejected():
    frame=pd.DataFrame([{'split_index':1,'activity':'a','variant':'attribute_x','support':50,
                         'top1_accuracy':.5,'class_f1':.5,'ovr_macro_f1':.5}])
    with pytest.raises(ValueError):paired_summary(frame,['attribute_x'])
