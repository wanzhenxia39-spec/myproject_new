"""Contract tests for split lineage, independent binding and exact feature selection."""
from copy import deepcopy
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from zshar.features.catalog import feature_catalog, resolve_plan_features
from zshar.features.groups import expand_feature_groups, load_groups
from zshar.semantics import planner
from zshar.semantics.completion import build_prompt as completion_prompt
from zshar.semantics.refinement import bind_features, validate_bound_plan, validate_concepts


def concept(name):
    return {'name': name, 'definition': f'Abstract physical factor {name}',
            'levels': {'Low': 'small degree', 'Medium': 'moderate degree', 'High': 'large degree'}}


def keep_plan(dataset='pamap2', index=3):
    return {'schema_version': '2.0', 'dataset': dataset, 'split_index': index,
            'refinement_needed': False, 'decision_reason': 'No supported additional distinctions.',
            'attributes': [concept('cadence')],
            'operations': [{'action': 'keep', 'inputs': ['cadence'], 'outputs': ['cadence'],
                            'reason': 'Rate remains useful.'}],
            'semantic_redundancy_review': 'Only one concept; no duplicate semantics.'}


def context(dataset='pamap2', index=3, a1_names=None):
    return dict(dataset=dataset, split_index=index, a1_names=a1_names or ['cadence'],
                seen=['walking', 'running'], unseen=['cycling', 'jumping'])


def binding(name='cadence', groups=None, selectors=None):
    groups = groups or ['G6']
    return {'attribute': name, 'primary_groups': groups, 'auxiliary_groups': [],
            'group_evidence': {g: 'Periodic timing supplies evidence of repetition rate.' for g in groups},
            'feature_selectors': selectors or {'positions': [], 'modalities': [], 'axes': []}}


def bound(plan=None, **kwargs):
    return bind_features(plan or keep_plan(), {'attribute_bindings': [binding(**kwargs)]})


def gap_row(index=3):
    return dict(split_index=index, seen_classes='walking|running', unseen_classes='cycling|jumping',
                D_kNN=.5, D_kNN_per_unseen_json=json.dumps({'cycling': .6, 'jumping': .4}),
                nearest_seen_topk_json=json.dumps({'cycling': [['walking', .6]], 'jumping': [['running', .4]]}))


def inputs(tmp_path):
    a1 = tmp_path / 'a1.json'
    a1.write_text(json.dumps({k: {'cadence': 'Medium'} for k in ['walking', 'running', 'cycling', 'jumping']}))
    row = gap_row()
    row['a1_sha256'] = hashlib.sha256(a1.read_bytes()).hexdigest()
    gap = tmp_path / 'gap.csv'
    pd.DataFrame([row]).to_csv(gap, index=False)
    return a1, gap


@pytest.mark.parametrize('dataset', ['pamap2', 'mhealth', 'dsads', 'uschad', 'wisdm', 'ucihar', 'shoaib', 'hhar'])
def test_bound_columns_for_all_configured_sensors(dataset):
    names = expand_feature_groups(dataset)['feature_names']
    plan = bound(keep_plan(dataset))
    result = validate_bound_plan(plan, feature_names=names, **context(dataset))
    cols = result['attributes']['cadence']
    assert cols['dimension'] > 0
    assert [names[i] for i in cols['indices']] == cols['feature_names']
    assert all(result['feature_metadata'][name]['group'] == 'G6' for name in cols['feature_names'])


def test_operations_cover_split_merge_drop_and_keep():
    plan = keep_plan()
    plan['attributes'] = [concept(n) for n in ['cadence', 'sustained_change', 'abrupt_change', 'posture_dynamics']]
    plan['operations'] += [
        {'action': 'split', 'inputs': ['smoothness'], 'outputs': ['sustained_change', 'abrupt_change']},
        {'action': 'merge', 'inputs': ['posture_variability', 'posture_inclination'], 'outputs': ['posture_dynamics']},
        {'action': 'drop', 'inputs': ['impulsiveness'], 'outputs': []}]
    for op in plan['operations'][1:]:
        op.update(reason='Separate temporal change factors, avoid overlapping impact semantics.',
                  target_unseen=['cycling'], contrast_unseen=['jumping'], seen_anchors=['walking'])
    plan['refinement_needed'] = True
    ctx = context(a1_names=['cadence', 'smoothness', 'posture_variability', 'posture_inclination', 'impulsiveness'])
    validate_concepts(plan, **ctx)
    plan['operations'][2]['inputs'].append('smoothness')
    with pytest.raises(ValueError, match='exactly once'):
        validate_concepts(plan, **ctx)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(refinement_needed=True),
    lambda p: p.update(split_index=99),
    lambda p: p['attributes'][0].update(name='ankle_acc_x_cadence'),
    lambda p: p['attributes'][0].update(body_position='ankle'),
    lambda p: p['attributes'][0]['levels'].pop('High'),
    lambda p: p['operations'][0].update(inputs=['unknown']),
    lambda p: p['operations'][0].update(action='split'),
    lambda p: p['operations'][0].update(seen_anchors=['cycling']),
    lambda p: p.update(semantic_redundancy_review=''),
])
def test_invalid_concept_contract(mutation):
    plan = keep_plan()
    mutation(plan)
    with pytest.raises(ValueError):
        validate_concepts(plan, **context())


def test_count_bounds_and_exact_duplicate_semantics():
    with pytest.raises(ValueError, match='count'):
        validate_concepts(keep_plan(), min_attributes=2, **context())
    plan = keep_plan()
    plan['attributes'].append({**concept('cadence'), 'name': 'repetition_rate'})
    with pytest.raises(ValueError, match='Duplicate'):
        validate_concepts(plan, **context())


@pytest.mark.parametrize('mutation', [
    lambda b: b['attribute_bindings'][0].update(definition='changed meaning'),
    lambda b: b.update(attribute_bindings=[]),
    lambda b: b['attribute_bindings'].append(binding()),
    lambda b: b['attribute_bindings'][0].update(attribute=[]),
])
def test_binding_cannot_edit_or_omit_concepts(mutation):
    response = {'attribute_bindings': [binding()]}
    mutation(response)
    with pytest.raises(ValueError):
        bind_features(keep_plan(), response)


@pytest.mark.parametrize('mutation', [
    lambda a: a.update(primary_groups=[]),
    lambda a: a.update(primary_groups=['G15']),
    lambda a: a.update(auxiliary_groups=['G6']),
    lambda a: a.update(group_evidence={}),
    lambda a: a.update(feature_selectors={'positions': ['missing_sensor']}),
    lambda a: a.update(feature_selectors={'axes': ['mag']}),
])
def test_invalid_bindings(mutation):
    plan = bound()
    mutation(plan['attributes'][0])
    with pytest.raises(ValueError):
        validate_bound_plan(plan, feature_names=['hand_acc_x_fft_dom_freq'], **context())


def test_reject_group_without_actual_columns_and_empty_filter():
    with pytest.raises(ValueError, match='no available'):
        validate_bound_plan(bound(), feature_names=['hand_acc_x_mean'], **context())
    with pytest.raises(ValueError, match='no available'):
        validate_bound_plan(bound(selectors={'positions': ['ankle']}),
                            feature_names=['hand_acc_x_fft_dom_freq'], **context())


def test_derived_tags_and_combined_filters_preserve_order():
    names = ['hand_CAGH', 'hand_AVG', 'hand_AVH', 'hand_EVA_1', 'hand_acc_gyro_ratio',
             'hand_acc_mag_fft_dom_freq', 'hand_acc_x_fft_dom_freq', 'hand_ARATG',
             'hand_acc_z_grav_angle_std', 'hand_GYRO_Z_ENERGY_phys']
    tags = feature_catalog(names, dataset='pamap2')
    assert tags['hand_CAGH']['axes'] == list('xyz')
    assert tags['hand_EVA_1']['axes'] == list('xyz')
    assert tags['hand_AVG']['axes'] == ['x']
    assert tags['hand_AVH']['axes'] == ['y', 'z']
    assert tags['hand_ARATG']['modalities'] == ['gyro']
    assert tags['hand_acc_gyro_ratio']['modalities'] == ['acc', 'gyro']
    assert tags['hand_acc_z_grav_angle_std']['axes'] == list('xyz')
    assert tags['hand_GYRO_Z_ENERGY_phys']['axes'] == ['z']
    plan = bound(groups=['G13', 'G6'], selectors={'positions': ['hand'], 'axes': ['x'], 'modalities': ['acc']})
    result = resolve_plan_features(plan, names, dataset='pamap2')['attributes']['cadence']
    assert result['indices'] == [1, 6]
    assert result['feature_names'] == ['hand_AVG', 'hand_acc_x_fft_dom_freq']


def test_prompts_remove_old_constraints_and_separate_binding(tmp_path):
    a1, gap = inputs(tmp_path)
    info = planner.resolve_split_from_gap_csv(gap, None, None, 3)
    prompt = planner.build_prompt('pamap2', info['seen'], info['unseen'], json.loads(a1.read_text()), info, {})
    assert 'Initial attribute set A1:' not in prompt
    assert '4. Redundancy' not in prompt
    assert 'body position/body part' not in prompt
    assert 'nearest_seen_topk' in prompt and 'NOT semantic redundancy' in prompt
    assert 'OTHER UNSEEN' in prompt and 'keep' in prompt and 'merge' in prompt
    binding_text = planner.build_binding_prompt(keep_plan(), ['hand_acc_x_fft_dom_freq'], load_groups(), {})
    assert 'including kept' in binding_text and 'primary_groups' in binding_text
    plan = bound(selectors={'positions': ['ankle']})
    completed = completion_prompt('pamap2', ['walking'], plan)
    assert 'ankle' not in completed
    assert 'primary_groups' not in completed and 'group_evidence' not in completed
    assert 'statistically correlated' in completed


def test_split_lookup_rejects_mismatch_duplicate_and_stale_a1(tmp_path):
    a1, gap = inputs(tmp_path)
    with pytest.raises(ValueError, match='disagree'):
        planner.resolve_split_from_gap_csv(gap, ['running'], None, 3)
    info = planner.resolve_split_from_gap_csv(gap, None, None, 3)
    a1.write_text(a1.read_text() + '\n')
    with pytest.raises(ValueError, match='different A1'):
        planner.run_split(dataset='pamap2', a1_path=a1, gap_path=gap, gap_info=info,
                          output_path=tmp_path/'plan.json', feature_names=['hand_acc_x_fft_dom_freq'],
                          feature_source='fixture', config=load_groups(), metadata={}, dry_run=True)
    pd.DataFrame([gap_row(), gap_row()]).to_csv(gap, index=False)
    with pytest.raises(ValueError, match='unique'):
        planner.resolve_split_from_gap_csv(gap, None, None, 3)


def test_two_live_stages_use_new_binding_and_export_mapping(tmp_path, monkeypatch):
    a1, gap = inputs(tmp_path)
    info = planner.resolve_split_from_gap_csv(gap, None, None, 3)
    calls = []
    response = {'attribute_bindings': [binding(groups=['G11'])]}
    def fake_llm(prompt, model, temperature):
        calls.append(prompt)
        return json.dumps(keep_plan() if len(calls) == 1 else response)
    monkeypatch.setattr(planner, 'call_gemini', fake_llm)
    result = planner.run_split(dataset='pamap2', a1_path=a1, gap_path=gap, gap_info=info,
                              output_path=tmp_path/'plan.json', feature_names=['hand_acc_x_pe'],
                              feature_source='fixture', config=load_groups(), metadata={})
    assert len(calls) == 2
    final = json.loads((tmp_path/'plan.json').read_text())
    assert final['attributes'][0]['primary_groups'] == ['G11']  # No inherited G6 mapping.
    mapping = json.loads(open(result['feature_mapping']).read())
    assert mapping['attributes']['cadence']['indices'] == [0]
    assert mapping['_meta']['split_index'] == 3


def test_batch_cli_offline_and_prompt_preview(tmp_path, monkeypatch):
    a1, gap = inputs(tmp_path)
    pd.DataFrame([gap_row(3), gap_row(7)]).to_csv(gap, index=False)
    names = tmp_path/'names.npy'
    np.save(names, ['hand_acc_x_fft_dom_freq'])
    root = tmp_path/'responses'
    for index in [3, 7]:
        folder = root/f'split_{index}'
        folder.mkdir(parents=True)
        (folder/'concepts.json').write_text(json.dumps(keep_plan(index=index)))
        (folder/'bindings.json').write_text(json.dumps({'attribute_bindings': [binding()]}))
    def no_live(*args, **kwargs):
        pytest.fail('Offline/dry-run must not contact the LLM')
    monkeypatch.setattr(planner, 'call_gemini', no_live)
    base = ['planner', '--dataset-name', 'pamap2', '--a1-json', str(a1), '--dknn-csv', str(gap),
            '--feature-names-npy', str(names), '--all-splits']
    monkeypatch.setattr(sys, 'argv', base + ['--output-root', str(tmp_path/'preview'), '--dry-run'])
    planner.main()
    assert not list((tmp_path/'preview').rglob('a2_attribute_plan.json'))
    monkeypatch.setattr(sys, 'argv', base + ['--output-root', str(tmp_path/'out'), '--responses-root', str(root)])
    planner.main()
    manifest = json.loads((tmp_path/'out/manifest.json').read_text())
    assert [entry['split_index'] for entry in manifest] == [3, 7]
    for entry in manifest:
        assert json.loads(open(entry['plan']).read())['split_index'] == entry['split_index']


def test_resume_saved_concepts_in_place_without_another_concept_call(tmp_path, monkeypatch):
    a1, gap = inputs(tmp_path)
    saved = tmp_path/'plan.concepts.json'
    saved.write_text(json.dumps(keep_plan()))
    before = saved.read_bytes()
    calls = []
    def fake_llm(prompt, *args):
        calls.append(prompt)
        return json.dumps({'attribute_bindings': [binding()]})
    monkeypatch.setattr(planner, 'call_gemini', fake_llm)
    planner.run_split(dataset='pamap2', a1_path=a1, gap_path=gap,
                      gap_info=planner.resolve_split_from_gap_csv(gap, None, None, 3),
                      output_path=tmp_path/'plan.json', feature_names=['hand_acc_x_fft_dom_freq'],
                      feature_source='fixture', config=load_groups(), metadata={}, concepts_path=saved)
    assert len(calls) == 1 and 'Frozen concepts' in calls[0]
    assert saved.read_bytes() == before


def test_real_columns_required_for_final_validation():
    with pytest.raises(ValueError, match='Actual feature names'):
        validate_bound_plan(bound(), **context())
