"""CPU-only publication plot from audited frozen scores; no model inference.
Run: uv run --no-project --with matplotlib==3.11.2 python figures/er_evidence_gap_20260915/plot_evidence_gap.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'reports/er_correction_cpu_20260914'
OUT=Path(__file__).resolve().parent
assert json.loads((DATA/'FINAL_COMPLETED.json').read_text())['audit_passed']
rows=[json.loads(l) for l in (DATA/'precision_sensitivity/scores.jsonl').read_text().splitlines()]
core=sorted([r for r in rows if r['boundary']=='native' and r['group']=='sufficient_failure'],key=lambda r:r['review_id'])
assert len(core)==6
alias=json.loads((DATA/'precision_sensitivity/alias_sensitivity.json').read_text())
assert alias['post_hoc'] and alias['review_id']=='E017'
methods=['hidden','observed','er_1','entropy_matched']
full_margin=lambda r,m:r['scores']['gold'][m]['full_sum']-r['scores']['original'][m]['full_sum']
export=[]
for r in core:
 d=r['divergence']['log_odds'];h,o=d['hidden'],d['observed']
 export.append(dict(case=r['review_id'],historical_parity=r['checks']['historical_reproduction_pass'],hidden=h,observed=o,delta=o-h,er=d['er_1'],positive_but_insufficient=h<o<0))
with (OUT/'plotted_cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=export[0].keys());w.writeheader();w.writerows(export)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'stix'})
blue='#0072B2';gray='#666666';green='#009E73';purple='#8860A0'
fig,axes=plt.subplots(1,2,figsize=(7.2,3.55),gridspec_kw={'width_ratios':[1.16,1]})
fig.subplots_adjust(left=.105,right=.985,bottom=.23,top=.90,wspace=.35)
a,b=axes
a.axhline(0,color='0.35',lw=.8);a.axvline(0,color='0.35',lw=.8)
a.plot([-15,15],[-15,15],color='0.6',ls='--',lw=.9,zorder=0)
offsets={'E002':(7,-8),'E003':(-31,11),'E004':(-15,8),'E008':(-8,12),'E017':(6,-14),'E030':(8,-7)}
for r in export:
 a.scatter(r['hidden'],r['observed'],s=42,facecolor=blue if r['historical_parity'] else 'white',edgecolor=blue,lw=1.35,zorder=4)
 a.annotate(r['case'],(r['hidden'],r['observed']),xytext=offsets[r['case']],textcoords='offset points',fontsize=8)
a.text(-13.8,-3.8,'Toward correct,\nbut still wrong\n'+r'$m_{\rm hid}<m_{\rm obs}<0$',fontsize=8,linespacing=1.4,va='top',bbox=dict(facecolor='white',edgecolor='none',pad=.3))
a.text(8.4,-10.5,r'$y=x$'+' : no shift',color=gray,fontsize=8)
a.set(xlim=(-15,16),ylim=(-12,16),xlabel='Evidence-hidden token margin\n(correct − wrong; nat)',ylabel='Evidence-visible token margin\n(correct − wrong; nat)')
a.set_xticks([-10,0,10]);a.set_yticks([-10,0,10])
a.legend(handles=[Line2D([],[],marker='o',ls='',color=blue,label='Historical check passed (n=3)',markersize=5),Line2D([],[],marker='o',ls='',color=blue,markerfacecolor='white',label='FP32 sensitivity only (n=3)',markersize=5)],loc='upper left',frameon=True,facecolor='white',edgecolor='white',framealpha=1,borderpad=.15,handletextpad=.45,fontsize=7.2)
a.text(-.18,1.02,'(a)',transform=a.transAxes,fontsize=10)
values=[full_margin(alias,m) for m in methods]
b.bar(range(4),values,color=[gray,blue,green,purple],width=.57,zorder=3)
b.axhline(0,color='0.35',lw=.8)
for i,v in enumerate(values):b.text(i,v+(.13 if v>=0 else -.13),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
b.set_xticks(range(4),['Evidence\nhidden','Evidence\nvisible','ER\n'+r'$\alpha=1$','Entropy\nmatched'])
b.tick_params(axis='x',length=0,pad=6)
b.set(ylim=(-4,3.5),ylabel='Full-answer log-probability margin\n(correct − wrong; nat)')
b.set_yticks([-4,-2,0,2])
prereg=next(r for r in core if r['review_id']=='E017')
fig.text(.792,.115,'E017: common-name alias (post hoc), n=1',ha='center',fontsize=7.3)
fig.text(.792,.07,'Preregistered full-name reference:',ha='center',fontsize=7.3)
fig.text(.792,.025,f"visible {full_margin(prereg,'observed'):+.2f}; ER {full_margin(prereg,'er_1'):+.2f} nat",ha='center',fontsize=7.3)
b.text(-.18,1.02,'(b)',transform=b.transAxes,fontsize=10)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'evidence_update_gap.{ext}',dpi=300,facecolor='white')
plt.close(fig)
# Full answer / formatting sensitivity stays reviewable; no favorable metric selection.
sensitivity=[]
for r in [next(r for r in core if r['review_id']=='E017'),alias,next(r for r in core if r['review_id']=='E004')]:
 for metric in ['full_sum','full_mean','answer_sum','answer_mean']:
  sensitivity.append(dict(case=r['review_id'],candidate='common-name (post hoc)' if r is alias else 'preregistered reference',metric=metric,**{m:r['scores']['gold'][m][metric]-r['scores']['original'][m][metric] for m in methods}))
with (OUT/'candidate_sensitivity.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=sensitivity[0].keys());w.writeheader();w.writerows(sensitivity)
paths=['FINAL_COMPLETED.json','AUDIT.json','precision_sensitivity/scores.jsonl','precision_sensitivity/alias_sensitivity.json']
manifest={'source_sha256':{p:hashlib.sha256((DATA/p).read_bytes()).hexdigest() for p in paths},'matplotlib':matplotlib.__version__,'n_core':6,'n_historical_pass':3,'panel_b_post_hoc':True,'no_new_model_inference':True,'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'cases':export,'alias_full_margins':dict(zip(methods,values))},indent=2))
