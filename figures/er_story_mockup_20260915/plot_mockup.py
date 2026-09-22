"""Scientific-figure-making style: SYNTHETIC layout mockup, never real results.
Run: uv run --no-project --with matplotlib==3.11.2 python figures/er_story_mockup_20260915/plot_mockup.py
"""
from pathlib import Path
import os,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'SYNTHETIC_DATA.json'
d=json.loads(DATA.read_text())
assert d['data_status']=='SYNTHETIC_LAYOUT_MOCKUP_NOT_EXPERIMENTAL_RESULTS'
PALETTE={'blue_main':'#0F4D92','blue_secondary':'#3775BA','green_1':'#DDF3DE','green_3':'#8BCF8B','red_1':'#F6CFCB','red_2':'#E9A6A1','red_strong':'#B64342','neutral':'#CFCECE','ink':'#272727','muted':'#767676'}
def apply_publication_style():
 plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':15,'axes.labelsize':14,'axes.linewidth':1.8,'xtick.labelsize':13,'ytick.labelsize':12,'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False,'legend.fontsize':12,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300,'text.color':PALETTE['ink'],'axes.labelcolor':PALETTE['ink']})
def finalize_figure(fig):
 for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'er_story_SYNTHETIC_mockup.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
apply_publication_style()
fig,axes=plt.subplots(1,3,figsize=(16.2,5.6))
fig.subplots_adjust(left=.055,right=.987,top=.72,bottom=.29,wspace=.39)
fig.text(.5,.943,'From useful evidence to correct supervision',ha='center',va='center',fontsize=24,fontweight='bold')
fig.text(.5,.877,'LAYOUT MOCKUP  •  ALL VALUES ARE SYNTHETIC  •  NOT EXPERIMENTAL RESULTS',ha='center',fontsize=12,color=PALETTE['red_strong'],fontweight='bold')
# Titles and small per-panel marks make cropped panels retain synthetic status.
titles=['(a) How does the problem evolve?','(b) Can the target cross the boundary?','(c) Does the student learn to correct?']
for ax,title in zip(axes,titles):
 ax.set_title(title,loc='center',fontsize=14,fontweight='bold',pad=35)
 ax.text(1,1.025,'SYNTHETIC',ha='right',transform=ax.transAxes,fontsize=8.5,color=PALETTE['red_strong'])
 ax.tick_params(length=4,width=1.25)
 ax.yaxis.grid(True,color='#E8E8E8',lw=.7)
 ax.set_axisbelow(True)
# A: three mutually exclusive state categories tracked at student checkpoints.
a=axes[0];x=np.asarray(d['panel_a']['steps'],dtype=float)
arr=np.asarray(d['panel_a']['values'],dtype=float)
assert arr.shape==(3,len(x)) and np.allclose(arr.sum(0),100) and np.all(np.diff(x)>0)
colors=[PALETTE['green_3'],PALETTE['blue_main'],PALETTE['muted']]
markers=['s','o','D'];styles=['--','-',':']
for j in [1,0,2]:
 vals=arr[j];label=d['panel_a']['categories'][j]
 a.plot(x,vals,color=colors[j],lw=3 if j==1 else 2,marker=markers[j],ms=6 if j==1 else 4.5,ls=styles[j],label=label)
 a.annotate(f'{vals[-1]:.0f}%',(x[-1],vals[-1]),xytext=(6,0),textcoords='offset points',va='center',fontsize=11,color=colors[j],fontweight='bold' if j==1 else 'normal')
a.set(ylim=(0,70),xlim=(x[0]-2,x[-1]+15),ylabel='Share of states (%)')
a.set_yticks([0,20,40,60]);a.set_xticks(x[::2])
a.set_xlabel('Student training step (HotpotQA)',fontsize=12,labelpad=8)
a.legend(loc='upper center',bbox_to_anchor=(.5,-.29),fontsize=10,ncol=1,handlelength=2,handletextpad=.5,labelspacing=.4)
# B: cohort selected by the original teacher; all branch endpoints are mock values.
b=axes[1];prefix=d['panel_b']['shared_prefix'];thr=d['panel_b']['decision_threshold']
b.axhline(thr,ls=(0,(4,3)),lw=1.25,color='#808080')
b.text(.03,thr+3,'Correct answer preferred',fontsize=10,color='#676767')
b.plot([0,1],prefix,color='#636363',lw=2.5,marker='o',ms=7,zorder=4)
b.plot([1,2],[prefix[1],d['panel_b']['er_target']],color=PALETTE['blue_main'],lw=3,marker='o',ms=7,zorder=4)
b.plot([1,2],[prefix[1],d['panel_b']['entropy_matched_control']],color=PALETTE['red_strong'],lw=2.1,marker='s',ms=6,ls='--',zorder=3)
for xx,v in [(0,prefix[0]),(1,prefix[1]),(2,d['panel_b']['er_target'])]:b.annotate(f'{v}%',(xx,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=13,fontweight='bold',color=PALETTE['blue_main'] if xx==2 else PALETTE['ink'])
b.annotate(f"{d['panel_b']['entropy_matched_control']}%",(2,d['panel_b']['entropy_matched_control']),xytext=(0,-20),textcoords='offset points',ha='center',fontsize=13,color=PALETTE['red_strong'])
b.set(ylim=(0,100),xlim=(-.15,2.2),ylabel='Correct-answer support (%)')
b.set_yticks([0,25,50,75,100]);b.set_xticks([0,1,2],d['panel_b']['conditions'],fontsize=11.5)
b.legend(handles=[Line2D([],[],color=PALETTE['blue_main'],lw=3,marker='o',label='Evidence-residual target'),Line2D([],[],color=PALETTE['red_strong'],lw=2,marker='s',ls='--',label='Confidence-only control')],loc='upper center',bbox_to_anchor=(.5,-.245),fontsize=10.5,handlelength=2)
# C: future paired student experiment; no artificial uncertainty intervals.
c=axes[2];vals=np.asarray(d['panel_c']['values'],dtype=float)
assert vals.shape==(3,)
cs=[PALETTE['red_1'],PALETTE['green_3'],PALETTE['blue_main']]
for j,(method,value,color) in enumerate(zip(d['panel_c']['methods'],vals,cs)):
 bars=c.bar(j,value,.55,color=color,edgecolor='#3C3C3C',linewidth=.85,hatch=['','..','//'][j])
 c.text(j,value+2,f'{value:.0f}',ha='center',va='bottom',fontsize=13,fontweight='bold' if j==2 else 'normal')
c.set(ylim=(0,100),ylabel='Student errors corrected (%)')
c.set_yticks([0,25,50,75,100]);c.set_xticks(np.arange(3),d['panel_c']['methods'])
c.set_xlabel('HotpotQA held-out probe set',fontsize=12,labelpad=8)
fig.text(.5,.022,'A proposed evidence chain: prevalence → target intervention → student correction.  No result in this draft has been measured.',ha='center',fontsize=10.7,color=PALETTE['muted'])
finalize_figure(fig)
(ROOT/'MOCKUP_MANIFEST.json').write_text(json.dumps({'data_status':d['data_status'],'data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'new_experiments_run':False,'error_bars':'none; no fabricated confidence intervals','formats':['pdf','svg','png'],'skill':'scientific-figure-making'},indent=2))
print('SYNTHETIC mockup saved; no experiment run or real result modified.')
