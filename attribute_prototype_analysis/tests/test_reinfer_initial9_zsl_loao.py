import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT=Path('/home/wanzhen/attribute_prototype_analysis/reinfer_initial9_zsl_loao.py')
spec=importlib.util.spec_from_file_location('fresh_loao',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_every_removal_reloads_only_retained_classifiers(tmp_path):
    attributes=[f'a{i}' for i in range(9)]
    manifest={'models':{a:{'path':a,'backend':'constant'} for a in attributes}}
    loaded=[];predicted=[]
    class FakePredictor:
        def __init__(self,name):self.name=name
        def predict_proba(self,frame):
            predicted.append(self.name)
            return pd.DataFrame({'Low':np.ones(len(frame))},index=frame.index)
    def loader(location,backend):
        loaded.append(location.name)
        return FakePredictor(location.name)
    frame=pd.DataFrame({'feature':[1.,2.]},index=[10,12])
    for removed in attributes:
        kept=[a for a in attributes if a!=removed]
        before=len(loaded)
        probability,orders,calls=module.fresh_attribute_probabilities(
            tmp_path,manifest,frame,['feature'],kept,loader)
        assert loaded[before:]==kept
        assert predicted[before:]==kept
        assert len(calls)==8 and probability.shape==(2,8,3)
        assert all(c['load_calls']==c['predict_proba_calls']==1 for c in calls)
    assert len(loaded)==len(predicted)==72


def test_identical_configurations_still_make_new_load_and_predict_calls(tmp_path):
    calls=[]
    class FakePredictor:
        def predict_proba(self,frame):
            calls.append('predict')
            return pd.DataFrame({'High':[1.]},index=frame.index)
    def loader(path,backend):
        calls.append('load');return FakePredictor()
    manifest={'models':{'a':{'path':'a','backend':'constant'}}}
    frame=pd.DataFrame({'x':[0.]})
    for _ in range(2):
        module.fresh_attribute_probabilities(tmp_path,manifest,frame,['x'],['a'],loader)
    assert calls==['load','predict','load','predict']
