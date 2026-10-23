from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
bold=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['svg.fonttype']='path'
fig,ax=plt.subplots(figsize=(18,22))
fig.patch.set_facecolor('#f5f7fb')
ax.set(xlim=(0,180),ylim=(220,0)); ax.axis('off')
fig.subplots_adjust(left=.015,right=.985,top=.995,bottom=.005)
ink='#172b45'; muted='#566780'
blue='#2563a6'; green='#177f6e'; purple='#7655a8'; amber='#a86818'

def text(x,y,s,size=11,color=ink,weight=False,ha='center'):
    ax.text(x,y,s,fontproperties=bold if weight else font,fontsize=size,color=color,
            ha=ha,va='center',linespacing=1.5,zorder=4)

def box(x,y,w,h,title,body='',color=blue,fill='white',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.3',
                 linewidth=1.15,edgecolor=color,facecolor=fill,zorder=2))
    if body:
        text(x+w/2,y+3.1,title,size,color,True)
        text(x+w/2,y+h/2+2,body,size-1)
    else:text(x+w/2,y+h/2,title,size,color,True)

def arrow(x1,y1,x2,y2,color=muted,dashed=False):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,
        linewidth=1.3,color=color,linestyle='--' if dashed else '-',zorder=1))

def panel(x,y,w,h,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.6,rounding_size=2',
          edgecolor='none',facecolor=color,zorder=0))

text(90,5,'从 IMU 输入到最终活动标签',25,weight=True)
text(90,11,'ZSHAR · 当前代码实现   |   Zero-shot / GZSL / 跨 subject',12,muted)
text(7,18,'01  传感器处理：所有场景共享',14,blue,True,ha='left')
for x,w,title,body in [
 (7,34,'IMU 原始数据','加速度 / 陀螺仪\n按数据集保留实际通道'),
 (48,37,'清洗、降采样、分窗','缺失值处理；定长窗口\nMHEALTH：20 Hz × 120点'),
 (92,39,'提取窗口特征','时域 / 频域 / STFT\njerk / ACF / 物理特征'),
 (138,35,'特征矩阵 X','保存特征列名和窗口索引\nMHEALTH：434列')]:box(x,23,w,15,title,body)
for a,b in [(41,48),(85,92),(131,138)]:arrow(a,30.5,b,30.5)

panel(6,43,168,57,'#eaf0fb')
text(10,47,'02  离线语义准备：ZSL / GZSL 按活动 split 生成 A2',14,blue,True,ha='left')
box(9,53,34,16,'初始 A1 语义原型','9属性定义 + 三级含义\nLLM 生成各活动 A1 向量',size=10.5)
box(49,53,38,16,'source-only 诊断','seen 类内部 pseudo-unseen\n筛选最多5张 issue cards',size=10.5)
box(93,53,36,16,'LLM 概念细化','A1 + issue cards + 组能力\nkeep / split / merge / drop',size=10.5)
box(135,53,36,16,'LLM 特征组绑定','参考 A1 映射软先验\nG1–G14 + 实际列可用性',size=10.5)
for a,b in [(43,49),(87,93),(129,135)]:arrow(a,61,b,61)
box(135,76,36,18,'审核与一次修复','10–14属性 + 有效非空绑定\nreject → replan / rebind\n仍未通过则回退 A1 九属性',size=10)
arrow(153,69,153,76)
box(90,76,39,18,'LLM 生成最终活动向量','最终属性定义 + 三级含义\n为全部活动补全原型\n输出 A2；fallback 为9维',size=10)
arrow(135,85,129,85)
box(49,76,35,18,'原型 P：供训练与匹配','ZSL/GZSL：当前 split A2\n未指定 A2：公共原型\n跨 subject：配置中的 A1',size=10)
arrow(90,85,84,85)
text(26,81,'refinement_evidence\n保存 pseudo D_kNN 等\n提示词只用筛选问题摘要',9,muted)
text(90,98,'绑定产物用于语义可观测性审核；当前训练入口尚未按 feature_mapping.json 为各属性裁剪特征。',9,amber)

text(7,106,'03  训练监督与模型：IMU 特征 → 属性等级',14,blue,True,ha='left')
box(7,112,52,17,'构造窗口级属性监督','训练窗口活动标签 y → 查原型 P(y)\n生成 Low / Medium / High 属性标签\nactivity_id、subject_id 不作为模型特征',size=11)
box(65,112,49,17,'按下方场景确定训练集','训练池内 70% / 30% 训练与验证\n条件允许时按活动分层\n模型输入仍是全部特征 X',size=11)
box(120,112,53,17,'训练并保存属性预测器','可选独立模型 / 多任务 / 分类器链\n验证报告用于评估及可选属性加权\n推理时只需窗口特征和已训练模型',size=11)
arrow(59,120,65,120);arrow(114,120,120,120)
arrow(66,94,66,102);arrow(66,102,33,102);arrow(33,102,33,112)
# Features bypass the semantic preparation along the right edge.
arrow(174.5,30.5,177,30.5);arrow(177,30.5,177,109);arrow(177,109,90,109);arrow(90,109,90,112)

panel(7,133,166,13,'#e9eef5')
text(10,136,'硬标签链示例',10,blue,True,ha='left')
text(96,137,'X → a1    →    (X, a1) → a2    →    …    →    (X, a1,…,a(k-1)) → ak',12,weight=True)
text(90,143,'训练后序属性时使用前序真实属性；测试时使用前序预测。ZSL/GZSL 可加载 split 对应正序、逆序等顺序。',9,muted)

text(7,152,'04  三种场景：测试窗口 → 属性预测 → 候选活动评分 → 标签',14,blue,True,ha='left')
lanes=[(7,blue,'#eaf2fc','Zero-shot · ZSL',
 '训练：seen 活动窗口\n测试：unseen 活动窗口\n人员不隔离；可出现相同 subject',
 '仅对 unseen 候选 U 评分\n预测属性与 U 的活动原型匹配',
 'Top-1 活动标签 ∈ U\n同时输出 Top-3 排名'),
 (64,green,'#e7f4ef','Generalized ZSL',
 '训练：seen，排除已抽作测试的窗口\n测试：seen + unseen 活动\n人员不隔离；按窗口索引防止重叠',
 '对全部候选 S ∪ U 评分\n可选 τ 温度缩放 + seen 分数减 γ\n或只对 seen 分数减 γ',
 'Top-1 活动标签 ∈ S ∪ U\n同时输出 Top-3 排名'),
 (121,purple,'#f0ebf8','跨 subject',
 '训练：train subjects 的全部活动\n测试：未参与训练的 test subjects\n当前 manifest 按人员 × 活动平衡测试',
 '对全部活动候选评分\n使用配置中的公共 A1 原型\n当前未接入 subject-specific A2',
 'Top-1 活动标签 ∈ 全部活动\n同时输出 Top-3 排名')]
for x,c,fill,title,split,match,result in lanes:
 panel(x,158,52,51,fill)
 text(x+26,161,title,13,c,True)
 box(x+1.5,165,49,13,'场景数据划分',split,color=c,size=9.8)
 box(x+1.5,181,49,7,'测试特征 X → 已训练模型 → 预测属性',color=c,size=9.3)
 box(x+1.5,191,49,14,'原型匹配 / 场景修正',match,color=c,size=9.8)
 arrow(x+26,178,x+26,181,c);arrow(x+26,188,x+26,191,c)
 box(x+1.5,211,49,8,result,color=c,fill=fill,size=10)
 arrow(x+26,205,x+26,211,c)
fig.savefig(OUT/'end_to_end_flow.png',dpi=170,facecolor=fig.get_facecolor())
fig.savefig(OUT/'end_to_end_flow.svg',facecolor=fig.get_facecolor())
fig.savefig(OUT/'end_to_end_flow.pdf',facecolor=fig.get_facecolor())
print(OUT/'end_to_end_flow.png')
