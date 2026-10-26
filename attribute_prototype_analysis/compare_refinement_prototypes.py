#!/usr/bin/env python3
"""Compare latest A1 v1.5 with all twenty A2 v4 splits, using class prototypes only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap
from datetime import datetime, timezone

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from attribute_discrimination_coverage import compute_attribute_coverage, read_vectors
from prototype_distance_heatmap import encode_prototypes, pairwise_prototype_distances

CODE_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = CODE_ROOT/'new_results'
NAMES = {'pamap2': 'PAMAP2', 'mhealth': 'MHEALTH', 'uschad': 'USC-HAD',
         'wisdm': 'WISDM', 'dsads': 'DSADS'}
METRICS = {'ordinal_probability': 'Normalized ordinal L1 distance',
           'euclidean': 'Normalized Euclidean distance (RMS)'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_dir(path):
    path = Path(path).expanduser().resolve()
    root = RESULT_ROOT.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Output must be within {RESULT_ROOT}')
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_dev != Path('/').stat().st_dev:
        raise ValueError('Output must be on the system filesystem')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def coverage_frame(vectors):
    return pd.DataFrame(compute_attribute_coverage(vectors)).set_index('attribute')


def coverage_across_splits(series):
    """Missing attributes remain NaN; means are conditional on presence."""
    by_split = pd.DataFrame(series)
    return by_split, by_split.mean(axis=1), by_split.count(axis=1), by_split.std(axis=1, ddof=0)


def distance_across_splits(matrices, activities):
    """Average distances, never ordinal labels or independently named coordinates."""
    aligned = [m.loc[activities, activities].to_numpy() for m in matrices]
    stack = np.stack(aligned)
    return (pd.DataFrame(stack.mean(0), index=activities, columns=activities),
            pd.DataFrame(stack.std(0), index=activities, columns=activities))


def _axis(ax, frame, title, coverage=False, counts=None, annotate=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    values = frame.to_numpy(dtype=float)
    rows, columns = values.shape
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#e5e7eb')
    norm = Normalize(0, 1)
    mesh = ax.pcolormesh(np.arange(columns+1)-.5, np.arange(rows+1)-.5,
                         np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                         rasterized=False, antialiased=False)
    ax.set_xlim(-.5, columns-.5); ax.set_ylim(rows-.5, -.5)
    ax.set_aspect('auto' if coverage else 'equal')
    def label(v, width):
        return textwrap.fill(str(v).replace('_', ' '), width=width, break_long_words=False)
    xlabels = [label(v, 17 if not coverage else 25) for v in frame.columns]
    ylabels = [label(v, 32 if coverage else 26) for v in frame.index]
    ax.set_xticks(np.arange(columns), xlabels, rotation=0 if coverage else 55,
                  ha='center' if coverage else 'right', fontsize=8)
    ax.set_yticks(np.arange(rows), ylabels, fontsize=8)
    ax.tick_params(which='both', length=0, pad=5)
    ax.set_xticks(np.arange(-.5, columns, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if not annotate:
                continue
            if np.isnan(value):
                text, color = 'N/A', '#64748b'
            else:
                text = f'{value:.2f}'
                if counts is not None:
                    text += f'\n(n={int(counts.iloc[i,j])})'
                r, g, b, _ = cmap(norm(value))
                color = 'white' if .2126*r+.7152*g+.0722*b < .52 else '#102a43'
            ax.text(j, i, text, ha='center', va='center', color=color,
                    fontsize=7 if columns <= 20 else 6)
    ax.set_title(title, fontsize=12, pad=12)
    return mesh


def plot_matrices(frames, titles, stem, label, dpi, coverage=False, counts=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(frames)
    rows = max(len(f) for f in frames)
    cols = max(len(f.columns) for f in frames)
    if coverage:
        width = max(9., 4.+cols*.68)*n
        height = max(5., 1.8+rows*.38)
    else:
        width = max(9., 3.+cols*.55)*n
        height = max(8., 3.+rows*.55)
    with plt.rc_context({'font.family':'DejaVu Sans', 'pdf.fonttype':42,
                         'ps.fonttype':42, 'svg.fonttype':'none'}):
        fig, axes = plt.subplots(1, n, figsize=(width, height), squeeze=False,
                                 layout='constrained')
        for i, (frame, title) in enumerate(zip(frames, titles)):
            mesh = _axis(axes[0,i], frame, title, coverage,
                         counts[i] if counts is not None else None)
        cb = fig.colorbar(mesh, ax=list(axes[0]), fraction=.022 if n>1 else .045,
                          pad=.025, ticks=np.linspace(0,1,6))
        cb.set_label(label)
        cb.outline.set_visible(False); cb.solids.set_rasterized(False)
        for ext in ['pdf','svg','png']:
            fig.savefig(Path(stem).with_suffix('.'+ext), dpi=dpi, facecolor='white')
        plt.close(fig)


def load_inputs(project, datasets):
    prepared, provenance = {}, []
    for ds in datasets:
        cfg_path = project/'configs/attrIbutes_v4'/ds/'experiment_config.json'
        cfg = json.loads(cfg_path.read_text())
        baseline = project/f'configs/attributes/activity_attribute_vectors_initial_{ds}_v1_5.json'
        expected = {Path(p).resolve(): h for p,h in cfg['input_files'].items()}
        if cfg['dataset'] != ds or cfg['seed'] != 42 or sha256(baseline) != expected[baseline.resolve()]:
            raise ValueError(f'A1 baseline does not match A2 generation provenance: {ds}')
        manifest_path = next(p for p in expected if p.name.endswith('_splits_seed42.json'))
        if sha256(manifest_path) != expected[manifest_path]:
            raise ValueError('Split manifest changed since A2 generation')
        split_manifest = json.loads(manifest_path.read_text())
        split_ids = sorted(s['split_index'] for s in split_manifest['splits'])
        actual_ids = sorted(int(p.parent.name.split('_')[1]) for p in
                            cfg_path.parent.glob('split_*/activity_attribute_vectors_a2.json'))
        if split_ids != actual_ids or len(split_ids) != 20:
            raise ValueError(f'Expected all twenty A2 splits: {ds}')
        paths = [(None, baseline)] + [(sid, cfg_path.parent/f'split_{sid}'/'activity_attribute_vectors_a2.json')
                                     for sid in split_ids]
        tables = []
        before_activities = None
        for sid, path in paths:
            vectors = read_vectors(path)
            table = encode_prototypes(vectors)
            if sid is None:
                before_activities = list(table.activities)
                if len(table.attributes) != 9:
                    raise ValueError('Expected nine-dimensional A1 v1.5')
            if set(table.activities) != set(before_activities):
                raise ValueError(f'Activity set differs between A1/A2: {path}')
            entry = {'dataset':ds,'stage':'before' if sid is None else 'after','split_index':sid,
                     'source_path':str(path.resolve()),'sha256':sha256(path),
                     'attributes':list(table.attributes),'activities':before_activities}
            if sid is not None:
                plan_path = path.parent/'a2_attribute_plan.json'
                plan = json.loads(plan_path.read_text())
                if {a['name'] for a in plan['attributes']} != set(table.attributes):
                    raise ValueError(f'A2 schema does not match refinement plan: {path}')
                entry.update(plan_path=str(plan_path.resolve()), plan_sha256=sha256(plan_path))
            tables.append((sid,vectors,table))
            provenance.append(entry)
        provenance.extend([{'dataset':ds,'stage':'configuration','source_path':str(p.resolve()),'sha256':sha256(p)}
                           for p in [cfg_path,manifest_path]])
        prepared[ds] = tables
    return prepared, provenance


def run(args):
    prepared, provenance = load_inputs(args.input_root.resolve(), args.datasets)
    root = output_dir(args.output_dir)
    if (root/'run_metadata.json').exists():
        raise FileExistsError('Use a fresh output directory to retain prior results')
    cache = output_dir(root/'_cache')
    for key,value in {'MPLCONFIGDIR':str(cache/'matplotlib'), 'XDG_CACHE_HOME':str(cache), 'TMPDIR':str(cache)}.items():
        os.environ[key] = value
    before_cov, after_cov, counts, std_cov = {}, {}, {}, {}
    records, coverage_records, checks = [], [], []
    for ds,tables in prepared.items():
        activities = list(tables[0][2].activities)
        baseline_attrs = list(tables[0][2].attributes)
        matrices = {metric:[] for metric in METRICS}
        split_coverage, split_summaries = {}, []
        for sid,vectors,table in tables:
            directory = output_dir(root/'before'/ds if sid is None else root/'after'/ds/'splits'/f'split_{sid}')
            write_json(directory/'vectors_snapshot.json',vectors)
            pd.DataFrame(table.values,index=table.activities,columns=table.attributes).loc[activities].to_csv(
                directory/'normalized_prototypes.csv',index_label='activity')
            coverage = coverage_frame(vectors)
            coverage.to_csv(directory/'attribute_coverage.csv')
            for attr,row in coverage.iterrows():
                coverage_records.append({'dataset':ds,'stage':'before' if sid is None else 'after',
                    'split_index':sid,'attribute':attr,**row.to_dict()})
            # Independent explicit pair enumeration verifies both reused metric functions.
            left,right = np.triu_indices(len(activities),1)
            values = table.values[[table.activities.index(a) for a in activities]]
            brute_cov = (values[left] != values[right]).mean(0)
            np.testing.assert_allclose(coverage.loc[list(table.attributes),'coverage'],brute_cov,atol=1e-14)
            for metric in METRICS:
                distance = pairwise_prototype_distances(table,metric=metric).loc[activities,activities]
                delta = np.abs(values[:,None,:]-values[None,:,:])
                expected = delta.mean(2) if metric=='ordinal_probability' else np.sqrt((delta**2).mean(2))
                np.testing.assert_allclose(distance,expected,atol=1e-14)
                np.testing.assert_allclose(distance,distance.T,atol=1e-14)
                assert np.all(np.diag(distance)==0) and np.all((distance.to_numpy()>=0)&(distance.to_numpy()<=1))
                filename = 'distance_matrix.csv' if metric=='ordinal_probability' else 'euclidean_distance_matrix.csv'
                distance.to_csv(directory/filename,index_label='activity')
                if sid is None:
                    plot_matrices([distance],[f'{NAMES[ds]}: before (A1 v1.5)'],
                        directory/('prototype_distance_heatmap' if metric=='ordinal_probability' else 'euclidean_distance_heatmap'),
                        METRICS[metric],args.dpi)
                else:
                    matrices[metric].append(distance)
            if sid is None:
                before_cov[ds] = coverage.coverage
            else:
                split_coverage[f'split_{sid}'] = coverage.coverage
                split_summaries.append({'split_index':sid,'attribute_count':len(table.attributes),
                    'mean_attribute_coverage':float(coverage.coverage.mean()),
                    **{metric+'_mean_distance':float(matrices[metric][-1].to_numpy()[left,right].mean()) for metric in METRICS}})
            checks.append({'dataset':ds,'split_index':sid,'stage':'before' if sid is None else 'after',
                           'independent_coverage_and_distances':'passed'})
        after_dir = output_dir(root/'after'/ds)
        by_split,after_cov[ds],counts[ds],std_cov[ds] = coverage_across_splits(split_coverage)
        by_split.to_csv(after_dir/'attribute_coverage_by_split.csv',index_label='attribute')
        by_split.notna().astype(int).to_csv(after_dir/'attribute_presence_by_split.csv',index_label='attribute')
        pd.DataFrame(split_summaries).to_csv(after_dir/'split_summary.csv',index=False)
        plot_matrices([by_split],[f'{NAMES[ds]}: A2 coverage by split (N/A = absent)'],
                      after_dir/'attribute_coverage_by_split', 'Attribute discrimination coverage', args.dpi,coverage=True)
        record={'dataset':ds,'class_count':len(activities),'split_count':20,'before_attribute_count':len(baseline_attrs),
            'after_attribute_count_min':min(r['attribute_count'] for r in split_summaries),
            'after_attribute_count_max':max(r['attribute_count'] for r in split_summaries),
            'before_mean_attribute_coverage':float(before_cov[ds].mean()),
            'after_mean_attribute_coverage':float(np.mean([r['mean_attribute_coverage'] for r in split_summaries]))}
        for metric in METRICS:
            mean,std = distance_across_splits(matrices[metric],activities)
            prefix='' if metric=='ordinal_probability' else 'euclidean_'
            baseline = pairwise_prototype_distances(tables[0][2],metric=metric).loc[activities,activities]
            mean.to_csv(after_dir/(prefix+'distance_matrix.csv'),index_label='activity')
            std.to_csv(after_dir/(prefix+'distance_matrix_std.csv'),index_label='activity')
            (mean-baseline).to_csv(after_dir/(prefix+'distance_change_matrix.csv'),index_label='activity')
            plot_matrices([mean],[f'{NAMES[ds]}: after (A2 mean of 20 splits)'],
                          after_dir/(prefix+'prototype_distance_heatmap'),METRICS[metric],args.dpi)
            comparison = output_dir(root/'comparison'/ds)
            plot_matrices([baseline,mean],[f'{NAMES[ds]}: before (A1 v1.5)',f'{NAMES[ds]}: after (A2 mean, n=20)'],
                          comparison/(prefix+'prototype_distance_before_after'),METRICS[metric],args.dpi)
            record.update({metric+'_before_mean_distance':float(baseline.to_numpy()[left,right].mean()),
                           metric+'_after_mean_distance':float(mean.to_numpy()[left,right].mean())})
        records.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    before = pd.DataFrame({NAMES[ds]:s for ds,s in before_cov.items()})
    after = pd.DataFrame({NAMES[ds]:s for ds,s in after_cov.items()})
    count = pd.DataFrame({NAMES[ds]:s for ds,s in counts.items()}).reindex(after.index).fillna(0).astype(int)
    std = pd.DataFrame({NAMES[ds]:s for ds,s in std_cov.items()}).reindex(after.index)
    for stage,matrix in [('before',before),('after',after)]:
        directory=output_dir(root/stage)
        matrix.to_csv(directory/'attribute_coverage_matrix.csv',index_label='attribute')
        if stage=='before':
            plot_matrices([matrix],['Before: A1 v1.5'],directory/'attribute_coverage_heatmap',
                          'Attribute discrimination coverage',args.dpi,coverage=True)
        else:
            count.to_csv(directory/'attribute_presence_counts.csv',index_label='attribute')
            std.to_csv(directory/'attribute_coverage_std.csv',index_label='attribute')
            for page,start in enumerate(range(0,len(matrix),24),1):
                chunk=matrix.iloc[start:start+24]
                plot_matrices([chunk],[f'After: A2 conditional mean, page {page} (n = splits containing attribute)'],
                    directory/f'attribute_coverage_heatmap_page_{page:02d}', 'Attribute discrimination coverage',args.dpi,
                    coverage=True,counts=[count.loc[chunk.index]])
    common = after.reindex(before.index)
    common_count = count.reindex(before.index)
    common.to_csv(root/'comparison'/'shared_attributes_after_coverage.csv',index_label='attribute')
    (common-before).to_csv(root/'comparison'/'shared_attributes_coverage_change.csv',index_label='attribute')
    plot_matrices([before,common],['Before: A1 v1.5','After: A2 mean on same attribute names'],
        root/'comparison'/'shared_attributes_coverage_before_after', 'Attribute discrimination coverage',args.dpi,
        coverage=True,counts=[pd.DataFrame(np.ones(before.shape),index=before.index,columns=before.columns),common_count])
    pd.DataFrame(records).to_csv(root/'dataset_summary.csv',index=False)
    pd.DataFrame(coverage_records).to_csv(root/'all_attribute_coverage.csv',index=False)
    for entry in provenance:
        if sha256(entry['source_path'])!=entry['sha256']:
            raise ValueError('Input changed during analysis')
        if 'plan_path' in entry and sha256(entry['plan_path'])!=entry['plan_sha256']:
            raise ValueError('Refinement plan changed during analysis')
    write_json(root/'run_metadata.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'output_dir':str(root),'inputs':provenance,'datasets':args.datasets,'dpi':args.dpi,
        'distance':'normalized L1 mean_k(abs(z_ik-z_jk)); z=Low:0, Medium:0.5, High:1',
        'euclidean':'normalized RMS sqrt(mean_k((z_ik-z_jk)^2))',
        'coverage':'count(i<j:p_ik!=p_jk)/binom(C,2)',
        'class_scope':'all activities; identical activity order before and after; no sample/model predictions',
        'distance_aggregation':'compute each split distance in its own K-dimensional schema, then mean/std across 20 splits; never average ordinal labels',
        'coverage_aggregation':'align exact attribute names; mean/std only over present splits; absent=NaN; counts exported; no synonym merging',
        'dataset_coverage_summary':'mean of each split mean across its actual attributes, then mean over splits',
        'color_limits':[0,1],'colormap':'Blues','missing_color':'gray; N/A',
        'source_sha256':{str(p):sha256(p) for p in [Path(__file__),CODE_ROOT/'prototype_distance_heatmap.py',CODE_ROOT/'attribute_discrimination_coverage.py']},
        'storage':{'realpath':str(root.resolve()),'device':root.stat().st_dev,'system_device':Path('/').stat().st_dev}})
    write_json(root/'verification_report.json',{'status':'passed','prototype_tables':len(checks),
        'refined_split_tables':len(checks)-len(prepared),'checks':checks,
        'input_hashes_unchanged':True,'same_activity_sets_and_order':True,
        'symmetric_distance_zero_diagonal_unit_range':True,'system_filesystem_verified':True})
    print(f'Completed: {root}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=Path('/home/wanzhen/myproject_new'))
    parser.add_argument('--output-dir',type=Path,default=RESULT_ROOT)
    parser.add_argument('--datasets',nargs='+',choices=list(NAMES),default=list(NAMES))
    parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args()
    if args.dpi<=0:
        parser.error('DPI must be positive')
    run(args)
