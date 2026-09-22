"""Readable Chinese redesign of the audited pilot; SOD-inspired panel layout.
Reproduce with: uv run --no-project --with matplotlib==3.11.2 python figures/er_evidence_gap_20260915/plot_readable_zh.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent
D=P.parents[1]/'reports/er_correction_cpu_20260914'
rows=[json.loads(x) for x in (D/'precision_sensitivity/scores.jsonl').read_text().splitlines()]
core={r['review_id']:r for r in rows if r['boundary']=='native' and r['group']=='sufficient_failure'}
alias=json.loads((D/'precision_sensitivity/alias_sensitivity.json').read_text())
assert len(core)==6 and alias['post_hoc']
fm.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
font=['DejaVu Sans',fm.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf').get_name()]
plt.rcParams.update({'font.family':font,'font.size':11,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
ink='#30383D';muted='#737B80';blue='#668CA8';green='#468A78';red='#BD7978';cream='#F4F0E8'
fig=plt.figure(figsize=(12,6.5),facecolor='white')
fig.text(.5,.955,'证据让判断更接近正确，但仍可能偏向错误答案',ha='center',va='center',fontsize=19,fontweight='bold',color=ink)
# User requested the boxed headers and soft panel styling of SOD.
for x,w,title in [(.025,.47,'（a）看到证据后，教师的判断怎样变化？'),(.52,.455,'（b）一个例子：谁更早获得诺贝尔奖？')]:
 fig.add_artist(FancyBboxPatch((x,.215),w,.64,boxstyle='round,pad=0.007,rounding_size=0.01',transform=fig.transFigure,fill=False,edgecolor='#92999D',linewidth=.9,linestyle=(0,(3.2,3.2)),zorder=0))
 fig.add_artist(Rectangle((x+.013,.835),w-.026,.052,transform=fig.transFigure,facecolor=cream,edgecolor='#6F7477',linewidth=.8,zorder=1))
 fig.text(x+w/2,.861,title,ha='center',va='center',fontsize=13,color=ink,zorder=2)
a=fig.add_axes([.145,.315,.32,.405]);b=fig.add_axes([.586,.315,.354,.405])
# All six fixed cases; names replace opaque identifiers. † marks precision failures.
ids=['E017','E004','E008','E002','E003','E030']
names={'E017':'诺贝尔奖先后','E004':'猎犬品种','E008':'乐队缩写含义','E002':'活动举办城市','E003':'两座城市共同点','E030':'希伯来文字类型'}
for ax in [a,b]:
 ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#BEC3C5')
 ax.tick_params(colors=muted,labelsize=10,length=0)
 ax.set_axisbelow(True)
# Discrete rows avoid the hidden-vs-visible scatterplot geometry.
a.axvspan(-15,0,color='#FAF3F1',zorder=0);a.axvspan(0,15,color='#F0F7F5',zorder=0)
a.axvline(0,color='#9AA2A5',lw=1,ls=(0,(3,3)))
for y,k in zip(range(5,-1,-1),ids):
 r=core[k];v=r['divergence']['log_odds'];h,o=v['hidden'],v['observed'];c=green if o>h else red
 a.annotate('',xy=(o,y),xytext=(h,y),arrowprops=dict(arrowstyle='->',color=c,lw=1.7,shrinkA=4,shrinkB=5,mutation_scale=10))
 a.scatter([h],[y],s=44,facecolor='white',edgecolor=c,lw=1.4,zorder=4)
 a.scatter([o],[y],s=44,facecolor=c,edgecolor='white',lw=.6,zorder=5)
 # Exact one-decimal labels make near-zero failures visible.
 a.text(h,y+.22,f'{h:+.1f}',ha='center',va='bottom',fontsize=9,color=muted)
 a.text(o,y-.23,f'{o:+.1f}',ha='center',va='top',fontsize=9,color=c)
a.set_xlim(-15,15);a.set_ylim(-.7,5.7);a.set_xticks([-10,0,10])
a.set_yticks(range(5,-1,-1),[names[k]+(' †' if not core[k]['checks']['historical_reproduction_pass'] else '') for k in ids])
a.tick_params(axis='y',labelcolor=ink,pad=10,labelsize=10)
a.set_xlabel('')
a.text(.24,1.02,'← 偏向错误',ha='center',transform=a.transAxes,color=red,fontsize=10)
a.text(.76,1.02,'偏向正确 →',ha='center',transform=a.transAxes,color=green,fontsize=10)
fig.legend(handles=[Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor=muted,label='隐藏检索内容',markersize=6),Line2D([],[],marker='o',ls='',color=muted,label='看到检索内容',markersize=6)],loc='center',bbox_to_anchor=(.263,.785),ncol=2,frameon=False,fontsize=10,columnspacing=1.5,handletextpad=.3)
# Full-answer common-name example: scores, not decoding or trained-student accuracy.
ms=['hidden','observed','er_1','entropy_matched']
margin=lambda r,m:r['scores']['gold'][m]['full_sum']-r['scores']['original'][m]['full_sum']
vals=[margin(alias,m) for m in ms]
b.axhspan(-4,0,color='#FAF3F1',zorder=0);b.axhspan(0,3.5,color='#F0F7F5',zorder=0)
b.axhline(0,color='#8E989B',lw=1)
b.bar(range(4),vals,width=.53,color=['#B7BEC2',blue,green,'#B8A9BE'],edgecolor='white',linewidth=.7,zorder=3)
for i,v in enumerate(vals):b.text(i,v+(.16 if v>=0 else -.15),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=12,fontweight='bold',color=ink)
b.set_xticks(range(4),['隐藏\n检索内容','看到\n检索内容','证据残差\n（我们的方法）','只调置信度\n（对照）'])
b.tick_params(axis='x',labelcolor=ink,pad=8,labelsize=10)
b.set_yticks([-4,-2,0,2]);b.set_ylim(-4,3.5);b.set_xlim(-.6,3.6)
b.set_ylabel('正确答案相对错误答案的支持分数',fontsize=10,labelpad=9,color=muted)
b.text(.03,.95,'更支持海明威（正确）',transform=b.transAxes,color=green,va='top',fontsize=10)
b.text(.97,.06,'更支持卡内蒂（错误）',transform=b.transAxes,color=red,ha='right',fontsize=10)
fig.text(.748,.789,'海明威 与 卡内蒂 · 常用姓名的补充分析',ha='center',fontsize=11,color=ink)
# Plain-language takeaways, scoped to cases rather than all rows.
fig.text(.26,.244,'前两例：向正确方向移动，却仍停在 0 左侧',ha='center',fontsize=11,color=ink)
fig.text(.747,.225,'这个候选对中，证据残差使支持分数越过了 0',ha='center',fontsize=11,color=ink)
fig.text(.03,.16,'读图：0 表示两者支持相当；负数偏向错误，正数偏向正确。左图看答案首次出现分歧的位置，右图比较完整答案。',fontsize=10,color=muted)
fig.text(.03,.117,'† 三例未通过历史概率复核，仅作同一高精度教师下的补充分析；六个固定案例全部保留，含反向变化。',fontsize=9.5,color=muted)
fig.text(.03,.074,f'右图为事后常用名分析。预定完整姓名未翻转：普通教师 {margin(core["E017"],"observed"):+.2f}，证据残差 {margin(core["E017"],"er_1"):+.2f}。',fontsize=9.5,color=muted)
fig.text(.03,.031,'这是教师候选偏好的小样本诊断，不代表实际生成正确率或学生训练收益；分数单位为自然对数。',fontsize=9.5,color=muted)
for ext in ['pdf','png','svg']:fig.savefig(P/f'evidence_update_gap_readable_zh.{ext}',dpi=220,facecolor='white')
plt.close(fig)
sources=['precision_sensitivity/scores.jsonl','precision_sensitivity/alias_sensitivity.json']
(P/'readable_zh_manifest.json').write_text(json.dumps({'source_sha256':{s:hashlib.sha256((D/s).read_bytes()).hexdigest() for s in sources},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'case_labels':names,'case_order':ids,'panel_b_full_answer_margins':dict(zip(ms,vals)),'n':6,'new_inference':False,'scope':'readability redesign; unchanged audited data'},ensure_ascii=False,indent=2))
print('Rendered readable Chinese PDF/PNG/SVG')
