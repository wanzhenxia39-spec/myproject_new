from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compare_refinement_prototypes import coverage_across_splits, distance_across_splits, output_dir
from prototype_distance_heatmap import encode_prototypes, pairwise_prototype_distances


def test_absent_attributes_are_not_zero_and_counts_are_explicit():
    matrix,mean,count,std=coverage_across_splits({
        'split_1':pd.Series({'common':.2,'added':1.}),
        'split_2':pd.Series({'common':.4}),
    })
    assert np.isnan(matrix.loc['added','split_2'])
    assert mean['added']==1. and count['added']==1 and std['added']==0.
    assert mean['common']==pytest.approx(.3) and count['common']==2
    assert std['common']==pytest.approx(.1)


@pytest.mark.parametrize('metric',['ordinal_probability','euclidean'])
def test_average_distances_not_prototypes(metric):
    first=encode_prototypes({'a':{'x':'Low','y':'High'},'b':{'x':'High','y':'Low'}})
    second=encode_prototypes({'b':{'x':'Low','y':'High'},'a':{'x':'High','y':'Low'}})
    matrices=[pairwise_prototype_distances(t,metric=metric) for t in [first,second]]
    mean,std=distance_across_splits(matrices,['a','b'])
    np.testing.assert_allclose(mean,[[0,1],[1,0]])
    np.testing.assert_allclose(std,0)
    # Averaging the raw prototypes would collapse both classes to (.5,.5).
    np.testing.assert_allclose((first.values+second.values[::-1])/2,np.full((2,2),.5))


def test_comparison_output_must_be_under_requested_system_directory():
    with pytest.raises(ValueError,match='within'):
        output_dir('/data/wanzhen/experiments/refinement_plots')
    with pytest.raises(ValueError,match='within'):
        output_dir('/home/wanzhen/elsewhere')
