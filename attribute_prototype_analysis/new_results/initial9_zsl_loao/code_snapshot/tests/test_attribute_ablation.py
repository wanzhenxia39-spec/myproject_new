import numpy as np
import pytest

from zshar.evaluation.attribute_ablation import (
    candidate_predictions, ordinal_distance_components, ordinal_scores,
)
from zshar.semantics.prototypes import match_prototype, probability_row
import pandas as pd


def test_components_match_reference_and_drop_renormalizes_weights():
    attrs = ["a", "b", "c"]
    labels = ["Low", "Medium", "High"]
    q = np.array([[[.1,.2,.7], [.6,.3,.1], [.2,.4,.4]]])
    prototypes = np.array([[0,1,2], [2,0,1]])
    weights = [1., 3., 2.]
    distances = ordinal_distance_components(q, prototypes)
    for removed in [None, 0, 1, 2]:
        scores = ordinal_scores(distances, weights, removed)
        kept = [a for i,a in enumerate(attrs) if i != removed]
        for c, proto in enumerate(prototypes):
            reference = match_prototype(
                predicted_values={}, prototype_values={a:labels[p] for a,p in zip(attrs,proto)},
                attribute_cols=kept, metric="ordinal_probability",
                attribute_weights=dict(zip(attrs,weights)),
                probability_distributions={a:dict(zip(labels, row)) for a,row in zip(attrs,q[0])},
                ordinal_values=dict(zip(labels,[0.,.5,1.])),
            )
            assert scores[0,c] == pytest.approx(reference.score, abs=1e-15)
    assert ordinal_scores(distances, weights, 0)[0,0] == pytest.approx(
        1-(3*distances[0,0,1]+2*distances[0,0,2])/5)


def test_removing_a_bad_coordinate_can_improve_predictions():
    # Attribute 0 is confidently wrong and has larger weight.
    q = np.array([[[0.,0.,1.], [1.,0.,0.]]])
    proto = np.array([[0,0],[2,2]])
    distance = ordinal_distance_components(q, proto)
    assert candidate_predictions(ordinal_scores(distance,[2,1]),["true","false"])[0] == "false"
    assert candidate_predictions(ordinal_scores(distance,[2,1],0),["true","false"])[0] == "true"


def test_absent_middle_state_does_not_compress_ordinal_scale():
    q=np.array([[[.25,0,.75]]])
    distance=ordinal_distance_components(q,np.array([[1]]),[[2,0]])
    assert distance[0,0,0] == .5


def test_native_normalization_matches_original_probability_row():
    q=np.array([[[0.,0.,1.0000001192092896]]])
    distance=ordinal_distance_components(q,np.array([[0],[2]]),[[2,0]])
    np.testing.assert_array_equal(distance[:, :, 0],[[1.,0.]])
    table=pd.DataFrame({"High":[q[0,0,2]],"Low":[0.]})
    assert probability_row({"a":table},"a",0)=={"High":1.,"Low":0.}


def test_constant_dimension_with_identical_prototypes_does_not_affect_ranking():
    q=np.array([[[.2,.3,.5], [0,0,1.]]])
    distance=ordinal_distance_components(q,np.array([[0,2],[2,2]]))
    np.testing.assert_array_equal(candidate_predictions(ordinal_scores(distance,[1,2]),["a","b"]),
                                  candidate_predictions(ordinal_scores(distance,[1,2],1),["a","b"]))


def test_tied_candidates_keep_original_tie_policy():
    scores=np.array([[.5,.5,.4,.5]])
    expected=np.array(["a","b","c","d"])[np.argsort(-scores,axis=1)[:,0]]
    np.testing.assert_array_equal(candidate_predictions(scores,["a","b","c","d"]),expected)


@pytest.mark.parametrize("weights,removed", [([1,0],None),([1,1],2),([float('nan'),1],None)])
def test_invalid_weights_and_dimensions_are_rejected(weights,removed):
    with pytest.raises(ValueError):ordinal_scores(np.zeros((1,2,2)),weights,removed)
