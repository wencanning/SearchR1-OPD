"""English, real-data-only training-step panels. Run in an isolated matplotlib environment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('summary'); a=p.parse_args()
src=Path(a.summary); d=json.loads(src.read_text())
if d.get('data_kind')!='REAL_EXPERIMENT': raise ValueError('Only actual experiment summaries are accepted')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
n=3 if d['C'] else 2
fig,axes=plt.subplots(1,n,figsize=(4.1*n,3.45),layout='constrained')
colors=['#3577A5','#D17842','#799B81','#9373A6']
def line(ax,rr,label,color):
    rr=sorted([r for r in rr if r['mean'] is not None],key=lambda r:r['step'])
    if not rr: return
    x=[r['step'] for r in rr]
    ax.plot(x,[100*r['mean'] for r in rr],label=label,color=color,lw=2,marker='o',ms=4)
    ax.fill_between(x,[100*r['low'] for r in rr],[100*r['high'] for r in rr],color=color,alpha=.12,lw=0)
for metric,col in zip(['Teacher favors correct','Evidence helps, still wrong','No helpful update'],colors):
    line(axes[0],[r for r in d['A'] if r['epsilon']==0 and r['metric']==metric],metric,col)
axes[0].set_title('(a) Evidence response during training',loc='left',fontweight='bold')
axes[0].set_ylabel('Share of eligible states (%)')
for target,label,col in zip(['hidden','observed','er','temperature'],['Evidence hidden','Original teacher','ER target','Entropy-matched teacher'],[colors[3],colors[0],colors[1],colors[2]]):
    line(axes[1],[r for r in d['B'] if r['target']==target],label,col)
axes[1].axhline(50,color='#AAAAAA',ls='--',lw=1)
axes[1].set_title('(b) When evidence helps, but not enough',loc='left',fontweight='bold')
axes[1].set_ylabel('Pairwise correct-answer support (%)')
if n==3:
    for m,label,col in zip(['opd','sod','er'],['OPD + GRPO','SOD + GRPO','ER-OPD + GRPO'],colors):
        line(axes[2],[r for r in d['C'] if r['method']==m and r['metric']=='Correction of initial errors'],label,col)
    axes[2].set_title('(c) Student correction on held-out questions',loc='left',fontweight='bold')
    axes[2].set_ylabel('Initial errors corrected: exact match (%)')
for ax in axes:
    ax.set_xlabel('Student training step'); ax.set_ylim(-2,102)
    ax.grid(axis='y',color='#E4E7EB',lw=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize=8,loc='best')
if d['label']=='pilot':
    fig.suptitle('Pilot diagnosis: legacy OPD without GRPO; not a matched method comparison',fontsize=10,color='#775C43')
for ext in ('png','pdf','svg'): fig.savefig(src.parent/f'evidence_gap_REAL.{ext}',dpi=300,bbox_inches='tight')
print(src.parent/'evidence_gap_REAL.pdf')
