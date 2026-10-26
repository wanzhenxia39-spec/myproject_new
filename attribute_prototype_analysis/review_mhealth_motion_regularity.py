#!/usr/bin/env python3
"""Version MHEALTH regularity annotations and recompute prototype-only coverage."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=Path('/home/wanzhen/myproject_new')
RESULTS=ROOT/'results/attribute_coverage_attributes_new'
DATASETS=['pamap2','mhealth','uschad','wisdm','dsads']
NAMES={'pamap2':'PAMAP2','mhealth':'MHEALTH','uschad':'USC-HAD','wisdm':'WISDM','dsads':'DSADS'}
ATTR='motion_regularity'
SOURCE_URL='https://archive.ics.uci.edu/dataset/319/mhealth+dataset'
# This is an explicit semantic review, NOT a rule inferred from IMU/test scores.
REVIEW={
    'standing still':('High','静止姿态的时间模式稳定、可预测；规律性不等同于存在周期运动。'),
    'sitting and relaxing':('High','保持坐姿的时间模式总体稳定；不因运动强度低而降低规律性。'),
    'lying down':('High','保持卧姿的时间模式稳定、可预测。'),
    'walking':('High','持续交替步态采用较稳定的重复时间模式。'),
    'climbing stairs':('High','连续上楼的重复步态采用较稳定的时间模式，不预设停顿或转向。'),
    'waist bends forward':('Medium','分次弯腰—恢复的往返动作有明确结构，但各次节拍及端点转换允许中等变化。此为语义标注假设。'),
    'frontal elevation of arms':('Medium','分次抬臂—放下的往返动作有结构，但启动、回落和各次间隔允许中等变化。此为语义标注假设。'),
    'knees bending (crouching)':('Medium','分次屈膝—起身动作有结构，但下降、转折、上升及各次间隔允许中等变化。此为语义标注假设。'),
    'cycling':('High','持续踩踏采用重复且较稳定的周期时间模式。'),
    'jogging':('High','持续慢跑采用较稳定的重复步态时间模式。'),
    'running':('High','持续跑步采用重复步态；强冲击本身不等同于不规律。'),
    'jump front & back':('Medium','前后交替跳有结构，但方向转换、落地与再次起跳节拍允许中等变化。此为语义标注假设。'),
}

from attribute_discrimination_coverage import read_vectors,compute_attribute_coverage,build_coverage_matrix
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def write_new_or_identical(path,text):
    if path.exists():
        if path.read_bytes()!=text.encode('utf-8'):
            raise FileExistsError(f'Refusing to overwrite a different vector/config file: {path}')
    else:
        path.write_text(text,encoding='utf-8')


def review_vectors(original):
    if set(original)!=set(REVIEW) or len(original)!=12:
        raise ValueError('Expected exactly the original 12 MHEALTH activities')
    if {row[ATTR] for row in original.values()}!={'High'}:
        raise ValueError('Original regularity is no longer uniformly High; review inputs again')
    updated={activity:dict(values) for activity,values in original.items()}
    for activity,(level,_) in REVIEW.items():
        updated[activity][ATTR]=level
    for activity in original:
        if list(updated[activity])!=list(original[activity]) or len(updated[activity])!=9:
            raise ValueError('Attribute schema changed')
        if any(updated[activity][a]!=original[activity][a] for a in original[activity] if a!=ATTR):
            raise ValueError('A protected attribute changed')
    return updated


def independent_coverage(vectors,attribute):
    rows=list(vectors.values())
    pairs=list(combinations(rows,2))
    return sum(a[attribute]!=b[attribute] for a,b in pairs)/len(pairs)


def draw_heatmaps(before,after,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def panel(ax,frame,title,delta=False):
        values=frame.to_numpy(dtype=float)
        image=ax.imshow(values,aspect='auto',vmin=-1 if delta else 0,vmax=1,
                        cmap='RdBu' if delta else 'Blues')
        ax.set_xticks(range(5),[NAMES[x] for x in frame.columns])
        ax.set_yticks(range(9),[x.replace('_',' ') for x in frame.index])
        ax.set_title(title,pad=12)
        for i in range(9):
            for j in range(5):
                value=values[i,j]
                label=f'{100*value:+.2f}' if delta else f'{value:.3f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=9,
                        color='white' if (abs(value)>.6 if delta else value>.6) else '#102a43')
        ax.set_xticks(np.arange(-.5,5,1),minor=True)
        ax.set_yticks(np.arange(-.5,9,1),minor=True)
        ax.grid(which='minor',color='white',lw=.7)
        ax.tick_params(which='both',length=0)
        return image
    for kind in ['latest','comparison','change']:
        frames=[before,after] if kind=='comparison' else [after-before if kind=='change' else after]
        titles=['Original initial 9D','MHEALTH regularity semantic review'] if kind=='comparison' else [
            'Coverage change: revised minus original (pp)' if kind=='change' else 'Attribute discrimination coverage — revised initial 9D']
        fig,axes=plt.subplots(1,len(frames),figsize=(10*len(frames),6.5),squeeze=False,layout='constrained')
        for ax,frame,title in zip(axes[0],frames,titles):
            image=panel(ax,frame,title,kind=='change')
        bar=fig.colorbar(image,ax=list(axes[0]),fraction=.025,pad=.025)
        bar.set_label('Coverage change (ratio; cells show pp)' if kind=='change' else 'Coverage (0–1)')
        for ext in ['png','pdf','svg']:
            fig.savefig(output/f'attribute_coverage_{kind}_9x5.{ext}',dpi=240,facecolor='white')
        plt.close(fig)


def run(args):
    output=args.output.resolve()
    if not output.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Prototype analysis output must use the formal system results directory')
    parent=output
    while not parent.exists():parent=parent.parent
    if parent.stat().st_dev!=Path('/').stat().st_dev or parent.stat().st_dev==Path('/data').stat().st_dev:
        raise ValueError('Prototype analysis output must be on the actual system filesystem')
    output.mkdir(parents=True,exist_ok=True)
    (output/'cache/matplotlib').mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(output/'cache/matplotlib'))
    source=PROJECT/'configs/attributes'
    target=PROJECT/'configs/attributes_new'
    target.mkdir(parents=True,exist_ok=True)
    old={ds:read_vectors(source/f'activity_attribute_vectors_initial_{ds}_v1_5.json') for ds in DATASETS}
    source_hashes={str((source/f'activity_attribute_vectors_initial_{ds}_v1_5.json').resolve()):
                   sha(source/f'activity_attribute_vectors_initial_{ds}_v1_5.json') for ds in DATASETS}
    schema=source/'initial_attributes.json'
    source_hashes[str(schema.resolve())]=sha(schema)
    new={ds:({a:dict(v) for a,v in old[ds].items()}) for ds in DATASETS}
    new['mhealth']=review_vectors(old['mhealth'])
    planned={}
    for ds in DATASETS:
        filename=f'activity_attribute_vectors_initial_{ds}_v1_5.json'
        text=json_text(new[ds]) if ds=='mhealth' else (source/filename).read_text(encoding='utf-8')
        planned[target/filename]=text
    planned[target/'initial_attributes.json']=schema.read_text(encoding='utf-8')
    # Validate ALL target names before writing any vector file.
    for path,text in planned.items():
        if path.exists() and path.read_bytes()!=text.encode('utf-8'):
            raise FileExistsError(f'Refusing to replace existing different input: {path}')
    for path,text in planned.items():write_new_or_identical(path,text)
    changes=[{'activity':a,'attribute':ATTR,'old_value':old['mhealth'][a][ATTR],
              'new_value':new['mhealth'][a][ATTR],'changed':old['mhealth'][a][ATTR]!=new['mhealth'][a][ATTR],
              'rationale':REVIEW[a][1]} for a in old['mhealth']]
    annotation={
        'revision':'initial9_mhealth_motion_regularity_semantic_review_v1',
        'dataset':'mhealth','modified_attribute':ATTR,'source_files_sha256':source_hashes,
        'original_definition':json.loads(schema.read_text())['attributes'][-1],
        'method':'explicit_manual_semantic_reannotation; no LLM API call, no IMU inspection, no model/test-score optimization',
        'source_reference':SOURCE_URL,
        'source_supported_facts':'UCI lists waist bends, frontal arm elevation, knees bending and front/back jumps as 20-repetition activities; the other eight use duration-based recording.',
        'inference_limit':'Repetition count does not establish temporal regularity. Medium labels are a reviewed semantic prior, not measured rhythm, observed pauses, or ground truth verified from IMU.',
        'operational_rule':'High: stable postures or sustained relatively uniform cycles. Medium: structured discrete return movements permitting moderate timing/phase variation. Low is reserved for poorly predictable patterns and is not forced into these scripted activities.',
        'level_histogram_before':dict(Counter(r[ATTR] for r in old['mhealth'].values())),
        'level_histogram_after':dict(Counter(r[ATTR] for r in new['mhealth'].values())),
        'activity_annotations':changes,
        'note':'Original inputs and runtime dataset configurations are not modified; prior predictors are not retrained or re-evaluated.'}
    write_new_or_identical(target/'motion_regularity_revision.json',json_text(annotation))
    readme='# 初始9维属性：MHEALTH motion_regularity 单属性语义重标注\n\n'
    readme+='本目录五份活动原型沿用 configs/attributes 的 initial v1.5。只有 MHEALTH 四个活动的 motion_regularity 从 High 改为 Medium；其余八维、其他活动/数据集及 initial_attributes.json 不变。\n\n'
    readme+='motion_regularity_revision.json 记录每个活动的判断、原始输入哈希、操作性标注规则与证据边界。Medium 是语义先验，不是实测时间规律性。未调用LLM、未读取IMU、未使用分类测试分数优化标签，也未改动运行配置或已保存模型。\n\n'
    readme+=f'覆盖率结果：{output}\n'
    write_new_or_identical(target/'README.md',readme)
    matrices={label:pd.DataFrame(build_coverage_matrix(v)['values'],
        index=build_coverage_matrix(v)['attributes'],columns=DATASETS) for label,v in [('original',old),('revised',new)]}
    before,after=matrices['original'],matrices['revised']
    all_metrics=[]
    for revision,vectors in [('original',old),('revised',new)]:
        for ds in DATASETS:
            for record in compute_attribute_coverage(vectors[ds]):
                if abs(record['coverage']-independent_coverage(vectors[ds],record['attribute']))>1e-15:
                    raise ValueError('Coverage differs from independent pair enumeration')
                all_metrics.append({'revision':revision,'dataset':ds,**record})
    old_saved=pd.read_csv(ROOT/'results/attribute_coverage_a1_v1_5/initial_a1_v1_5/attribute_coverage_matrix.csv',index_col=0)
    old_saved.columns=DATASETS
    if not np.allclose(before,old_saved.loc[before.index],atol=1e-15,rtol=0):
        raise ValueError('Original baseline no longer reproduces the previous coverage matrix')
    different=(after-before).abs()>1e-15
    if int(different.to_numpy().sum())!=1 or not different.loc[ATTR,'mhealth']:
        raise ValueError('Unexpected changed coverage cells')
    for label,frame in [('original',before),('latest',after),('change',after-before)]:
        frame.to_csv(output/f'attribute_coverage_{label}_9x5.csv',index_label='attribute')
        (100*frame).to_csv(output/f'attribute_coverage_{label}_percent_9x5.csv',index_label='attribute')
    metrics=pd.DataFrame(all_metrics)
    metrics.to_csv(output/'all_attribute_pair_counts.csv',index=False)
    annotation_table=pd.DataFrame(changes)
    annotation_table.to_csv(output/'mhealth_motion_regularity_annotations.csv',index=False)
    means=pd.DataFrame({'original_dataset_mean':before.mean(axis=1),'latest_dataset_mean':after.mean(axis=1),
                        'change_pp':100*(after-before).mean(axis=1)})
    means.to_csv(output/'attribute_means_across_5datasets.csv',index_label='attribute')
    sys.path.insert(0,str(ROOT/'new_results/initial9_seen_predictability/cache/python_deps'))
    with pd.ExcelWriter(output/'attribute_coverage_updated_9x5.xlsx',engine='openpyxl') as writer:
        for label,frame in [('latest_9x5',after),('original_9x5',before),('change_pp_9x5',100*(after-before))]:
            frame.to_excel(writer,sheet_name=label,index_label='attribute')
        annotation_table.to_excel(writer,sheet_name='mhealth_annotations',index=False)
        metrics.to_excel(writer,sheet_name='exact_pair_counts',index=False)
        means.to_excel(writer,sheet_name='attribute_dataset_means',index_label='attribute')
        for sheet in writer.sheets.values():sheet.freeze_panes='B2';sheet.auto_filter.ref=sheet.dimensions
    draw_heatmaps(before,after,output)
    report=['# MHEALTH motion_regularity 重标注及五数据集覆盖率更新','',
        '原始 initial v1.5 MHEALTH 的12个活动均为 High，因而该维不同等级活动对为0/66。',
        '新版保持属性名称、定义、Low/Medium/High等级说明和其余八维不变，仅将四类分次往返动作标为Medium：waist bends forward、frontal elevation of arms、knees bending (crouching)、jump front & back。其余八类为High；不为了增加覆盖率强行加入Low。',
        f'[UCI活动及采集协议]({SOURCE_URL}) 支持四类动作各采集20次这一事实，但不直接决定其时间规律性。Medium依据分次动作允许中等节拍/相位变化的操作性语义规则，是人工审阅的先验假设；没有实测节拍、观察停顿或重新计算属性可预测性。静止状态的时间模式可稳定预测，因此不机械地标为Low。',
        '原始配置、运行配置、模型、历史分类/可预测性结果保持不变。本次没有LLM请求、IMU推理、训练或测试分数调优。','',
        '## 属性区分覆盖率','',
        'R_k = 不同属性等级的无序活动对数 / C(C−1)/2。只判断等级是否不同，不按等级距离加权；与分类准确率、可预测性和属性互补性不同。本次覆盖全部活动原型，不基于20个split作模型评价。',
        'MHEALTH 新版等级计数：High=8、Medium=4、Low=0。不同等级活动对=8×4=32；总活动对=12×11/2=66。因此 R=32/66=0.484848（48.4848%），原值为0。',
        '五数据集9×5矩阵中仅 MHEALTH × motion_regularity 一个单元格变化；该属性跨5数据集的等权平均增加9.6970个百分点。',
        '覆盖率上升是更改语义原型的直接结果，不证明新标注更符合真实IMU，也不证明分类性能或属性可预测性提高。若后续采用新目标标签训练，该属性需要重新训练和验证；旧常量分类器的结果不能直接套用。','',
        '## 最新9×5矩阵（0–1）','',
        '|属性|'+'|'.join(NAMES[ds] for ds in DATASETS)+'|','|---|'+'---:|'*5]
    for attr,row in after.iterrows():report.append('|'+attr+'|'+'|'.join(f'{v:.6f}' for v in row)+'|')
    report+=['','## MHEALTH逐活动标注','', '|活动|原值|新版|理由|','|---|---|---|---|']
    for change in changes:report.append('|'+change['activity']+'|'+change['old_value']+'|'+change['new_value']+'|'+change['rationale']+'|')
    report+=['','## 结果入口','',
        '- attribute_coverage_latest_9x5.png/pdf/svg：最新矩阵图。',
        '- attribute_coverage_comparison_9x5.png/pdf/svg：原版/新版并排对照。',
        '- attribute_coverage_change_9x5.png/pdf/svg：新版减原版，格内标注百分点。',
        '- attribute_coverage_updated_9x5.xlsx：矩阵、差值、逐活动标注、精确配对计数及跨数据集均值。',
        '- 新版五份向量及定义文件位于 `/home/wanzhen/myproject_new/configs/attributes_new/`。该路径指向数据盘上的项目目录；本组分析结果本身保存在系统盘。','']
    (output/'REPORT_zh.md').write_text('\n'.join(report),encoding='utf-8')
    verification={'passed':True,'source_vector_files_unchanged':all(sha(p)==h for p,h in source_hashes.items()),
        'mhealth_changed_activity_count':sum(r['changed'] for r in changes),'changed_attribute':ATTR,
        'mhealth_other_8_attributes_identical':True,
        'other_4_dataset_files_byte_identical':all(sha(source/f'activity_attribute_vectors_initial_{ds}_v1_5.json')==sha(target/f'activity_attribute_vectors_initial_{ds}_v1_5.json') for ds in DATASETS if ds!='mhealth'),
        'attribute_definition_file_byte_identical':sha(schema)==sha(target/'initial_attributes.json'),
        'coverage_matrix_changed_cells':int(different.to_numpy().sum()),'independent_pair_enumeration_cells_checked':90,
        'mhealth_regularity_before':float(before.loc[ATTR,'mhealth']),
        'mhealth_regularity_after':float(after.loc[ATTR,'mhealth']),
        'source_files_sha256':source_hashes,'new_vector_files_sha256':{str(p.resolve()):sha(p) for p in planned},
        'semantic_prior_not_empirically_validated':True}
    if not verification['source_vector_files_unchanged'] or not verification['other_4_dataset_files_byte_identical']:
        raise ValueError('Protected inputs changed')
    write_new_or_identical(output/'input_manifest.json',json_text({'source':source_hashes,'revised':verification['new_vector_files_sha256']}))
    (output/'verification.json').write_text(json_text(verification),encoding='utf-8')
    snapshot=output/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__),ROOT/'attribute_discrimination_coverage.py',ROOT/'tests/test_mhealth_regularity_review.py']:
        if path.exists():shutil.copy2(path,snapshot/path.name)
    print(json.dumps({k:v for k,v in verification.items() if not k.endswith('sha256')},ensure_ascii=False,indent=2))
    print('Vectors:',target,'\nResults:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RESULTS)
    run(parser.parse_args())
