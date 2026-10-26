from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from review_mhealth_motion_regularity import review_vectors,independent_coverage,ATTR,PROJECT
from attribute_discrimination_coverage import compute_attribute_coverage


def original():
    return json.loads((PROJECT/'configs/attributes/activity_attribute_vectors_initial_mhealth_v1_5.json').read_text())


def test_revision_preserves_original_and_exactly_eight_other_dimensions():
    old=original();frozen=deepcopy(old);new=review_vectors(old)
    assert old==frozen
    changed=[]
    for activity in old:
        assert list(old[activity])==list(new[activity])
        for attribute in old[activity]:
            if old[activity][attribute]!=new[activity][attribute]:changed.append((activity,attribute))
    assert set(changed)=={(a,ATTR) for a in ['waist bends forward','frontal elevation of arms','knees bending (crouching)','jump front & back']}
    assert {new[a][ATTR] for a in ['standing still','sitting and relaxing','lying down']}=={'High'}


def test_coverage_changes_only_the_reviewed_dimension_and_matches_pair_enumeration():
    old=original();new=review_vectors(old)
    before={r['attribute']:r for r in compute_attribute_coverage(old)}
    after={r['attribute']:r for r in compute_attribute_coverage(new)}
    assert before[ATTR]['different_pairs']==0
    assert after[ATTR]['different_pairs']==32
    assert after[ATTR]['total_pairs']==66
    assert after[ATTR]['coverage']==pytest.approx(32/66)
    assert independent_coverage(new,ATTR)==after[ATTR]['coverage']
    assert {a for a in before if before[a]!=after[a]}=={ATTR}


def test_review_refuses_an_unexpected_preexisting_annotation():
    old=original();old['walking'][ATTR]='Medium'
    with pytest.raises(ValueError):review_vectors(old)
