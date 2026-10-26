#!/usr/bin/env python3
"""Per-activity LOAO analysis from saved predictions; no model loading/inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'new_results'
SOURCE=BASE/'initial9_zsl_loao'
ATTRIBUTES=['motion_intensity','cadence','smoothness','impulsiveness','posture_inclination',
            'posture_variability','rotational_motion','translational_motion','motion_regularity']
LABELS=['Intensity','Cadence','Smoothness','Impulsiveness','Inclination','Posture variability',
        'Rotational motion','Translational motion','Motion regularity']
METRICS=['top1_accuracy','class_f1','ovr_macro_f1']
NAMES={'pamap2':'PAMAP2','mhealth':'MHEALTH','uschad':'USC-HAD','wisdm':'WISDM','dsads':'DSADS'}

import numpy as np
import pandas as pd


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_metrics(truth,prediction,candidates):
    """F1 uses ALL classes in the split, retaining false positives for class c."""
    truth=np.asarray(truth,dtype=str)
    prediction=np.asarray(prediction,dtype=str)
    if len(truth)!=len(prediction) or not len(truth):
        raise ValueError('Missing/misaligned samples')
    if set(truth)!=set(candidates) or not set(prediction).issubset(candidates):
        raise ValueError('Labels/candidates differ from the Unseen candidate set')
    result=[]
    for activity in candidates:
        positive=truth==activity
        predicted=prediction==activity
        tp=int(np.sum(positive&predicted));fn=int(np.sum(positive&~predicted))
        fp=int(np.sum(~positive&predicted));tn=int(np.sum(~positive&~predicted))
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        negative_f1=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
        result.append({'activity':activity,'support':tp+fn,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'top1_accuracy':tp/(tp+fn),'class_f1':f1,'ovr_macro_f1':(f1+negative_f1)/2})
    return result


def paired_summary(metrics,attributes):
    full=metrics[metrics.variant=='full9'][['split_index','activity']+METRICS].rename(
        columns={m:'full_'+m for m in METRICS})
    drops=metrics[metrics.variant!='full9'].copy()
    paired=drops.merge(full,on=['split_index','activity'],how='left',validate='many_to_one')
    if paired[['full_'+m for m in METRICS]].isna().any().any():
        raise ValueError('Missing paired full baseline')
    for metric in METRICS:
        paired['delta_'+metric+'_pp']=100*(paired['full_'+metric]-paired[metric])
        paired['removed_minus_full_'+metric+'_pp']=-paired['delta_'+metric+'_pp']
    rows=[]
    for (activity,attribute),group in paired.groupby(['activity','variant'],sort=False):
        if attribute not in attributes or group.split_index.duplicated().any():
            raise ValueError('Unexpected attribute/duplicate split')
        row={'activity':activity,'removed_attribute':attribute,'n_unseen_splits':len(group),
             'unseen_split_indices':','.join(map(str,sorted(group.split_index))),
             'test_window_occurrences':int(group.support.sum())}
        for metric in METRICS:
            values=group['delta_'+metric+'_pp']
            row['full_'+metric+'_mean_pct']=float(100*group['full_'+metric].mean())
            row['removed_'+metric+'_mean_pct']=float(100*group[metric].mean())
            row['delta_'+metric+'_mean_pp']=float(values.mean())
            row['delta_'+metric+'_sd_pp']=float(values.std(ddof=1)) if len(values)>1 else np.nan
            row['removed_minus_full_'+metric+'_mean_pp']=-float(values.mean())
            row[metric+'_positive_splits']=int((values>1e-10).sum())
            row[metric+'_negative_splits']=int((values < -1e-10).sum())
            row[metric+'_zero_splits']=int((values.abs()<=1e-10).sum())
        rows.append(row)
    return paired,pd.DataFrame(rows)


def collect(source,dataset):
    directories=sorted((source/dataset).glob('split_*'))
    if len(directories)!=20:raise ValueError('Expected exactly 20 original ZSL splits')
    rows=[];coverage=[];audit={};overall=[]
    for directory in directories:
        split=int(directory.name.split('_')[1])
        saved=read_json(directory/'metrics.json')
        matching=read_json(directory/'matching.json')
        if matching['attributes']!=ATTRIBUTES:raise ValueError('Expected initial 9 attributes in original order')
        if saved['signature']['dataset']!=dataset or saved['signature']['split_index']!=split:
            raise ValueError('Dataset/split provenance mismatch')
        frame=pd.read_csv(directory/'activity_predictions.csv.gz')
        if frame.window_index.tolist()!=saved['signature']['test_window_indices'] or frame.window_index.duplicated().any():
            raise ValueError('Fixed test window IDs differ')
        candidates=matching['candidate_order']
        counts=frame.activity_name.value_counts()
        expected_count=50 if dataset=='pamap2' else 30
        if set(counts.index)!=set(candidates) or not counts.eq(expected_count).all():
            raise ValueError('Unseen test counts/candidates differ')
        baseline=frame['full9__prediction'].to_numpy(dtype=str)
        if not np.array_equal(baseline,frame['full9_prediction'].to_numpy(dtype=str)):
            raise ValueError('Duplicate saved baseline columns differ')
        historical={r['variant']:r for r in saved['rows']}
        if set(historical)!=set(['full9']+ATTRIBUTES):raise ValueError('Missing original variant metrics')
        for variant in ['full9']+ATTRIBUTES:
            predicted=frame[variant+'__prediction'].to_numpy(dtype=str)
            per_class=class_metrics(frame.activity_name,predicted,candidates)
            replay_accuracy=sum(r['tp'] for r in per_class)/len(frame)
            replay_macro_f1=float(np.mean([r['class_f1'] for r in per_class]))
            if abs(replay_accuracy-historical[variant]['top1_accuracy'])>1e-12 or abs(replay_macro_f1-historical[variant]['macro_f1'])>1e-12:
                raise ValueError('Per-class metrics do not reproduce saved split accuracy/Macro-F1')
            overall.append({'split_index':split,'variant':variant,'replayed_accuracy':replay_accuracy,
                            'replayed_macro_f1':replay_macro_f1,'matches_saved_metrics':True})
            rows.extend({'dataset':dataset,'split_index':split,'variant':variant,**r} for r in per_class)
        for activity in candidates:
            group=frame[frame.activity_name==activity]
            ids=group.activity_id.unique()
            if len(ids)!=1:raise ValueError('Activity IDs are ambiguous')
            coverage.append({'activity':activity,'activity_id':int(ids[0]),'split_index':split,
                'window_ids':list(map(int,group.window_index)),'support':len(group)})
        for name in ['metrics.json','matching.json','activity_predictions.csv.gz']:
            path=directory/name
            audit[str(path.resolve())]=digest(path)
    return pd.DataFrame(rows),coverage,audit,pd.DataFrame(overall)


def save_figure(fig,path):
    for ext in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+ext),dpi=240,bbox_inches='tight')


def plot_activity(summary,activities,output,dataset,second_metric):
    import matplotlib.pyplot as plt
    folder=output/('per_activity' if second_metric=='class_f1' else 'per_activity_ovr_macro_f1')
    folder.mkdir(exist_ok=True)
    second_label='Activity class F1' if second_metric=='class_f1' else 'One-vs-rest binary Macro-F1'
    for i,activity in enumerate(activities):
        group=summary[summary.activity==activity].set_index('removed_attribute').loc[ATTRIBUTES]
        fig,axes=plt.subplots(1,2,figsize=(13,5.7),sharey=True)
        for ax,metric,title in zip(axes,['top1_accuracy',second_metric],['Top-1 accuracy (class recall)',second_label]):
            values=group['delta_'+metric+'_mean_pp'].to_numpy()
            errors=group['delta_'+metric+'_sd_pp'].to_numpy()
            y=np.arange(9)
            ax.barh(y,values,height=.65,color=np.where(values < -1e-10,'#d88663','#6c9fd1'))
            mask=np.isfinite(errors)
            ax.errorbar(values[mask],y[mask],xerr=errors[mask],fmt='none',color='#444444',capsize=3,lw=.9)
            ax.axvline(0,color='black',lw=.8)
            ax.set_yticks(y,LABELS)
            ax.set_title(title)
            ax.set_xlabel('Full 9D minus removed 8D (percentage points)')
            extent=max(1.,float(np.max(np.abs(values)+np.nan_to_num(errors,nan=0))))
            ax.set_xlim(-1.2*extent,1.2*extent)
            ax.grid(axis='x',alpha=.2)
            for j,v in enumerate(values):
                ax.annotate(f'{v:+.2f}',(v,j),xytext=(4 if v>=0 else -4,5),textcoords='offset points',
                            ha='left' if v>=0 else 'right',fontsize=8)
        axes[0].invert_yaxis()
        n=int(group.n_unseen_splits.iloc[0])
        fig.suptitle(f'{NAMES[dataset]}: {activity} — Unseen in {n}/20 splits\n'
                     'Positive: removal hurts; negative: removal helps. Error bars: paired split SD.',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.9))
        safe=''.join(c if c.isalnum() else '_' for c in activity).strip('_')
        save_figure(fig,folder/f'{i+1:02d}_{safe}_impact')
        plt.close(fig)


def plot_overview(summary,activities,output,dataset,second_metric):
    """Horizontal bar glyphs give categorical activity columns and attribute rows."""
    import matplotlib.pyplot as plt
    second_label='Activity class F1' if second_metric=='class_f1' else 'One-vs-rest binary Macro-F1'
    fig,axes=plt.subplots(2,1,figsize=(21,11.5))
    for ax,metric,title in zip(axes,['top1_accuracy',second_metric],['Top-1 accuracy (class recall)',second_label]):
        matrix=summary.pivot(index='removed_attribute',columns='activity',values='delta_'+metric+'_mean_pp').loc[ATTRIBUTES,activities]
        limit=max(1.,float(np.abs(matrix.to_numpy()).max()))
        for column,activity in enumerate(activities):
            ax.axvline(column,color='#888888',lw=.6,zorder=1)
            ax.axvspan(column-.49,column+.49,color='#f6f6f6' if column%2==0 else 'white',zorder=0)
            for row,value in enumerate(matrix[activity]):
                width=.42*value/limit
                ax.barh(row,width,left=column,height=.52,color='#d88663' if value < -1e-10 else '#6c9fd1',zorder=2)
                ax.text(column,row-.30,f'{value:+.2f}',ha='center',va='bottom',fontsize=8,zorder=3)
        labels=[]
        for activity in activities:
            n=int(summary.loc[summary.activity==activity,'n_unseen_splits'].iloc[0])
            labels.append(textwrap.fill(activity,15)+f'\n(n={n}/20)')
        ax.set_xticks(np.arange(len(activities)),labels,fontsize=9)
        ax.set_yticks(np.arange(9),LABELS,fontsize=10)
        ax.set_xlim(-.5,len(activities)-.5);ax.set_ylim(8.6,-.7)
        ax.set_title(f'{title}: full minus removed (pp); shared bar scale within panel: ±{limit:.2f} pp',fontsize=12)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle(f'{NAMES[dataset]} initial 9D LOAO — activity-specific impact\n'
                 'Each cell: one signed horizontal bar; grey vertical line is zero. Values are mean paired differences.',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94),h_pad=2.5)
    name='activity_attribute_impact_top1_f1' if second_metric=='class_f1' else 'activity_attribute_impact_top1_ovr_macro_f1'
    save_figure(fig,output/name)
    plt.close(fig)


def write_report(output,dataset,summary,coverage,activities):
    lines=[f'# {NAMES[dataset]} 初始9维属性：按活动统计的逐维移除实验','',
        '本次只读取 initial9_zsl_loao 已保存的逐窗口真实活动标签及10种配置的预测，不加载模型、不重新预测、不重新训练或选模。使用初始 v1.5 属性及原有20个ZSL split，候选集合为每个 split 的 Unseen 类。',
        '主差值沿用原实验：完整9维 − 删除一维后的8维，单位为百分点（pp）。正值表示该维在当前条件下有正贡献；负值表示删除后更好。反方向“删除后 − 完整”同时保存在表格。','',
        '## 活动条件统计口径','',
        '对于活动 c，仅纳入它作为 Unseen 出现的 split 集合 S_c。先在每个 split 上分别计算完整/删除配置的指标并相减，再在同一 S_c 上等权求均值。不是将未出现该活动的 split 填0，也不是将所有活动均值无条件平均为原来的总体分数。',
        'Top-1(c)=TP/(TP+FN)，即真实属于 c 的测试窗口的识别准确率（类别召回率）；所有错判均进入 FN。',
        '单个活动的 F1(c)=2TP/(2TP+FP+FN)。FP 必须来自同一 split 中其他活动被预测成 c 的样本，因此先使用完整候选集合计算混淆计数，再提取该活动的 F1。不能仅筛选 y_true=c 的行后调用 macro F1。',
        '原来的 Macro-F1 是该 split 各 Unseen 活动 F1 的均值；主图第二面板显示它对应的活动类别 F1，而不将其误称为单个活动的跨类别 Macro-F1。为满足另一种“每活动宏F1”定义，另提供严格的一对其余（二分类）Macro-F1=(F1(c)+F1(non-c))/2 的双子图和表格；它包含其余类表现，不能与原多类 Macro-F1 混用。',
        f'本次 {NAMES[dataset]} 每个 split 的每个 Unseen 类固定'+('50' if dataset=='pamap2' else '30')+'个测试窗口。误差线为 S_c 上配对差值的样本标准差（ddof=1），不是置信区间；总览条形矩阵不画误差线，逐活动图和Excel提供标准差。n=1不计算标准差。split共享数据，因此不能将这些均值或标准差视为独立重复的显著性检验。',
        '完整/删除后的活动均值采用相同split集合，两者均值相减等于配对差值的均值。各活动的n不同，因此将12个活动条件均值再简单平均，不等于原20个split的总体均值；若要还原总体指标，应按各活动出现的split次数加权（本数据每split类别数与每类测试窗口数固定）。','',
        '## 每个活动实际作为 Unseen 的次数','',
        '|活动|n/20|窗口出现次数|split索引|','|---|---:|---:|---|']
    for activity in activities:
        row=coverage[coverage.activity==activity].iloc[0]
        lines.append(f'|{activity}|{row.n_unseen_splits}|{row.test_window_occurrences}|{row.unseen_split_indices}|')
    for metric,title in [('top1_accuracy','Top-1'),('class_f1','活动类别 F1')]:
        lines+=['',f'## {title}：完整9维 − 删除后的平均差值（pp）','',
                '|属性|'+'|'.join(activities)+'|','|---|'+'---:|'*len(activities)]
        for attr in ATTRIBUTES:
            g=summary[summary.removed_attribute==attr].set_index('activity')
            lines.append('|'+attr+'|'+'|'.join(f'{g.loc[a,"delta_"+metric+"_mean_pp"]:+.3f}' for a in activities)+'|')
    lines+=['','## 文件入口','',
        '- activity_attribute_impact_top1_f1.png/pdf/svg：纵轴9维属性、横轴活动类别的条形矩阵总览，上下两个面板分别为Top-1与活动类别F1。每列灰线为零点，条形向左/右表示负/正差值；各面板内部所有条形采用相同长度尺度。',
        f'- per_activity/：每个活动单独一张Top-1 / 类别F1双子图，共{len(activities)}张（PNG/PDF/SVG），带配对标准差。',
        '- activity_attribute_impact_top1_ovr_macro_f1.* 和 per_activity_ovr_macro_f1/：严格二分类Macro-F1替代口径，含义与主图类别F1不同。',
        '- activity_loao_results.xlsx：绝对值、双方向差值、标准差、活动覆盖次数、逐split分类混淆计数及总体重放核对。',
        '- activity_attribute_summary.csv：每活动×每属性的完整值、删除后值、差值、标准差与正负split数。',
        '- per_split_activity_metrics.csv / paired_activity_deltas.csv：仅该活动作为Unseen时的逐split结果与配对差值。',
        '- source_audit.json / verification.json：输入文件哈希和核验结果。','',
        '该实验评价冻结系统中的条件边际影响，不能将各单维删除收益相加，也不能依据测试消融选择属性子集后将同一测试集当作独立验证。','']
    (output/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    index=[f'# {NAMES[dataset]} 按活动统计的 LOAO 结果入口','',
        '差值 = 完整9维 − 删除后8维（pp）。主分析第二面板为活动类别F1；另提供一对其余二分类Macro-F1，两者口径不同。','',
        '- [Top-1＋类别F1：属性×活动总览](activity_attribute_impact_top1_f1.png) · [PDF](activity_attribute_impact_top1_f1.pdf)',
        '- [Top-1＋一对其余Macro-F1：总览](activity_attribute_impact_top1_ovr_macro_f1.png) · [PDF](activity_attribute_impact_top1_ovr_macro_f1.pdf)',
        '- [Excel完整对照表](activity_loao_results.xlsx)',
        '- [中文指标说明与全部差值矩阵](REPORT_zh.md)','',
        '|活动|作为Unseen的split数|Top-1＋类别F1|Top-1＋一对其余Macro-F1|','|---|---:|---|---|']
    for i,activity in enumerate(activities):
        safe=''.join(c if c.isalnum() else '_' for c in activity).strip('_')
        filename=f'{i+1:02d}_{safe}_impact'
        n=int(coverage.loc[coverage.activity==activity,'n_unseen_splits'].iloc[0])
        index.append(f'|{activity}|{n}/20|[PNG](per_activity/{filename}.png) · [PDF](per_activity/{filename}.pdf)|'
                     f'[PNG](per_activity_ovr_macro_f1/{filename}.png) · [PDF](per_activity_ovr_macro_f1/{filename}.pdf)|')
    index+=['','代码入口：`/home/wanzhen/attribute_prototype_analysis/analyze_initial9_loao_by_activity.py`。',
            '复现命令：','', '```bash',
            f'bash /home/wanzhen/attribute_prototype_analysis/run_initial9_loao_by_activity.sh {dataset}', '```','']
    (output/'RESULT_INDEX_zh.md').write_text('\n'.join(index),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=list(NAMES),default='pamap2')
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=(args.output or BASE/'initial9_zsl_loao_by_activity'/args.dataset).resolve()
    if not output.is_relative_to(BASE.resolve()):raise ValueError('Use an explicit system-disk new_results directory')
    parent=output
    while not parent.exists():parent=parent.parent
    if parent.stat().st_dev!=Path('/').stat().st_dev or parent.stat().st_dev==Path('/data').stat().st_dev:
        raise ValueError('Expected the actual system filesystem')
    output.mkdir(parents=True,exist_ok=True)
    for folder in ['tmp','pycache','matplotlib']:(output/'cache'/folder).mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(output/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    metrics,occurrences,audit,overall=collect(args.source,args.dataset)
    common_audit={str((args.source/'protocol.json').resolve()):digest(args.source/'protocol.json'),**audit}
    prior=output/'source_audit.json'
    if prior.exists() and read_json(prior)!=common_audit:raise ValueError('Source predictions changed since the last analysis')
    write_json(prior,common_audit)
    paired,summary=paired_summary(metrics,ATTRIBUTES)
    coverage_rows=[]
    for activity in metrics.activity.unique():
        records=[r for r in occurrences if r['activity']==activity]
        ids={r['activity_id'] for r in records}
        if len(ids)!=1:raise ValueError('Activity IDs disagree across splits')
        coverage_rows.append({'activity':activity,'activity_id':ids.pop(),'n_unseen_splits':len(records),
            'test_window_occurrences':sum(r['support'] for r in records),
            'unique_test_windows':len(set(w for r in records for w in r['window_ids'])),
            'unseen_split_indices':','.join(map(str,sorted(r['split_index'] for r in records)))})
    coverage=pd.DataFrame(coverage_rows).sort_values('activity_id').reset_index(drop=True)
    activities=coverage.activity.tolist()
    for name,frame in [('per_split_activity_metrics',metrics),('paired_activity_deltas',paired),
                       ('activity_attribute_summary',summary),('activity_unseen_coverage',coverage),('overall_metric_replay',overall)]:
        frame.to_csv(output/(name+'.csv'),index=False)
    matrices={}
    for metric in METRICS:
        for field in ['mean','sd']:
            matrix=summary.pivot(index='removed_attribute',columns='activity',values=f'delta_{metric}_{field}_pp').reindex(index=ATTRIBUTES,columns=activities)
            matrix.to_csv(output/f'{metric}_delta_{field}_pp_9x{len(activities)}.csv')
            matrices[metric+'_'+field]=matrix
    sys.path.insert(0,str(BASE/'initial9_seen_predictability/cache/python_deps'))
    with pd.ExcelWriter(output/'activity_loao_results.xlsx',engine='openpyxl') as writer:
        for name,frame in [('summary',summary),('coverage',coverage),('paired_deltas',paired),('per_split_class_metrics',metrics),('overall_metric_replay',overall)]:
            frame.to_excel(writer,sheet_name=name,index=False)
        for name,frame in matrices.items():frame.to_excel(writer,sheet_name=name)
        for sheet in writer.sheets.values():
            sheet.freeze_panes='B2';sheet.auto_filter.ref=sheet.dimensions
    for metric in ['class_f1','ovr_macro_f1']:
        plot_activity(summary,activities,output,args.dataset,metric)
        plot_overview(summary,activities,output,args.dataset,metric)
    write_report(output,args.dataset,summary,coverage,activities)
    snapshot=output/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for file in [Path(__file__),ROOT/'test_initial9_loao_by_activity.py',ROOT/'run_initial9_loao_by_activity.sh']:
        if file.exists():shutil.copy2(file,snapshot/file.name)
    verification={'passed':True,'dataset':args.dataset,'source_splits':20,'activities':len(activities),
        'attribute_activity_cells':len(summary),'unseen_class_split_occurrences':len(occurrences),
        'per_split_class_variant_rows':len(metrics),'paired_drop_class_split_rows':len(paired),
        'replayed_overall_variant_metrics':len(overall),'new_model_loads':0,'new_inference_calls':0,
        'delta':'full9_minus_removed8_percentage_points','primary_second_panel':'per_class_f1',
        'supplementary_second_panel':'one_vs_rest_binary_macro_f1',
        'all_input_hashes_unchanged_after_analysis':all(digest(p)==h for p,h in common_audit.items()),
        'code_sha256':{str(p):digest(p) for p in snapshot.iterdir()}}
    if not verification['all_input_hashes_unchanged_after_analysis']:raise ValueError('Inputs changed during analysis')
    write_json(output/'verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print('Output:',output)


if __name__=='__main__':main()
