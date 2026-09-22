"""Plot real local validation records, without smoothing or extrapolation."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/er_baseline_curves_20260915'
d=json.loads((OUT/'comparison.json').read_text());records=list(csv.DictReader((OUT/'all_validation_points.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(13.5,3.9),layout='constrained')
colors=['#353F4B','#CE7346','#408E91','#8B6EAC']
styles=['-','-','--',':']
for rid,label,color,style in zip(d['primary_run_ids'],d['primary_run_ids'].values(),colors,styles):
 rr=sorted([r for r in records if r['run']==rid],key=lambda r:int(r['step']))
 for ax,key in zip(axes,['macro7','micro512','hotpotqa']):
  ax.plot([int(r['step']) for r in rr],[float(r[key])*100 for r in rr],color=color,linestyle=style,marker='o',lw=2,label=label)
for ax,title in zip(axes,['Reported Avg: 7-dataset macro mean','Overall accuracy: 512 questions','HotpotQA accuracy: 73 questions']):
 ax.set_title(title,fontsize=11,fontweight='bold',loc='left');ax.set_xlabel('Training step');ax.set_ylabel('Accuracy (%)')
 ax.set_xticks([50,100,150]);ax.set_xlim(45,155);ax.set_ylim(15,47);ax.grid(axis='y',alpha=.18)
 ax.axvspan(50,100,color='#8B9AA5',alpha=.06,zorder=-2)
axes[0].annotate('One Bamboogle question\ncontributes 14.29 pp to Avg',xy=(100,42.415),xytext=(53,44.3),fontsize=8,color='#79533D',arrowprops={'arrowstyle':'->','color':'#79533D','lw':.8})
axes[1].legend(loc='lower right',fontsize=8,frameon=False)
fig.suptitle('0.5B OPD vs ER-OPD | Matched curve means use steps 50 and 100 only',fontsize=12)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'validation_curves.{ext}',dpi=220,bbox_inches='tight')
print(OUT/'validation_curves.png')
