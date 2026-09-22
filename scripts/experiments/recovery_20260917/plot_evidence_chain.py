"""English measured-data figure. Refuses to plot incomplete training outcomes."""
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR','/tmp/recovery_20260917_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/'reports/recovery_20260917';OUT=ROOT/'figures/recovery_20260917'


def main():
    update=json.loads((DATA/'update_summary.json').read_text());rec=json.loads((DATA/'recovery_summary.json').read_text())
    paths=[DATA/'probe_teacher/answers.jsonl',DATA/'recovery_summary.json',DATA/'update_summary.json']
    teacher=[json.loads(l) for l in paths[0].read_text().splitlines()]
    # Inspect keys explicitly rather than inferring output schema.
    assert teacher and 'variant' in teacher[0]
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12,'axes.labelsize':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300})
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.75),gridspec_kw={'width_ratios':[1.08,.98,1.24]})
    grey='#CFCECE';green='#AADCA9';blue='#3775BA';pink='#E9A6A1'
    records=[];rng=np.random.default_rng(2026091720)
    def ci(rows):
        qids=sorted({r['qid'] for r in rows});m=np.array([np.mean([r['em'] for r in rows if r['qid']==q]) for q in qids])
        low,high=np.quantile(m[rng.integers(0,len(m),size=(20000,len(m)))].mean(1),[.025,.975])*100
        return float(low),float(high)
    def rate_bars(ax,groups,colors):
        for x,(name,rs) in enumerate(groups):
            k=sum(r['em'] for r in rs);n=len(rs);y=100*k/n;lo,hi=ci(rs)
            ax.bar(x,y,width=.65,color=colors[x],edgecolor='#4D4D4D',linewidth=.8,zorder=2)
            ax.errorbar(x,y,yerr=np.array([[max(0,y-lo)],[max(0,hi-y)]]),fmt='none',color='#4D4D4D',capsize=3,lw=1,zorder=3)
            ax.text(x,min(hi+3,96),f'{k}/{n}',ha='center',va='bottom',fontsize=10)
            records.append(dict(panel=ax.get_title(),label=name.replace('\n',' '),correct=k,n=n,percent=y,low=lo,high=hi))
        ax.set_xticks(range(len(groups)),[x[0] for x in groups]);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Answer accuracy (%)')
    a=axes[0];a.set_title('(a) Testing the original explanation',loc='left',pad=14)
    variants=['actual_native','actual_neutral','support_native'];labels=['Original\ncontext','Last reasoning\nremoved','Annotated\nsupport*']
    rate_bars(a,[(label,[r for r in teacher if r['variant']==v]) for label,v in zip(labels,variants)],[grey,green,blue])
    a.text(.02,.97,'Teacher · 16 questions',transform=a.transAxes,va='top',fontsize=9,color='#666666')
    b=axes[1];b.set_title('(b) Do recovery paths exist?',loc='left',pad=14)
    vals=[rec['n_eligible_train'],rec['observed_student_all_failed'],rec['teacher_rescued_questions']]
    for x,(v,c) in enumerate(zip(vals,[grey,pink,blue])):
        b.bar(x,v,width=.65,color=c,edgecolor='#4D4D4D',linewidth=.8,zorder=2);b.text(x,v+.5,str(v),ha='center',fontsize=11)
    b.set_xticks([0,1,2],['All tested\nstates','Student failed\nboth attempts','Teacher\nrecovered']);b.set_ylabel('Number of questions');b.set_ylim(0,19);b.set_yticks([0,4,8,12,16])
    b.text(.02,.97,'Same state · same extra budget',transform=b.transAxes,va='top',fontsize=9,color='#666666')
    c=axes[2];c.set_title('(c) Do students learn from them?',loc='left',pad=14)
    arm_names=['zero','base','student_replay','teacher_repair'];arm_labels=['Before\nupdate','OPD\nupdate','+ Student\nreplay','+ Teacher\nrecovery']
    groups=[]
    for arm,label in zip(arm_names,arm_labels):
        p=DATA/('eval_'+arm)/'continuations.jsonl';paths.append(p)
        assert json.loads(p.with_name('status.json').read_text())['state']=='complete'
        groups.append((label,[json.loads(l) for l in p.read_text().splitlines()]))
    rate_bars(c,groups,[grey,pink,green,blue]);c.text(.02,.97,'Student · 8 held-out questions',transform=c.transAxes,va='top',fontsize=9,color='#666666')
    for ax in axes:
        ax.yaxis.grid(True,color='#E9E9E9',linewidth=.7,zorder=0);ax.set_axisbelow(True);ax.tick_params(axis='x',length=0,pad=7)
    fig.text(.052,.016,'Exploratory pilot. *Annotated support is diagnostic only. Error bars: question-bootstrap 95% intervals. Panel (c): 8 local updates, unequal extra compute.',fontsize=8.5,color='#555555')
    fig.tight_layout(rect=(0,.07,1,1),pad=1.3,w_pad=2.0)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/('evidence_chain.'+ext),facecolor='white',bbox_inches='tight')
    with (OUT/'source_values.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    (OUT/'caption.md').write_text('Exploratory evidence chain on HotpotQA. (a) Conditional teacher answering on16 input-selected support-covered questions, two samples each; removing only the latest interpretation differs from replacing observations with annotated supporting sentences. (b) Of16 new training states,12 have two student EM failures and4 of these admit at least one teacher EM success within the same additional token/search cap. Counts are nested subsets, not independent arms. (c) Fresh continuations from8 question-disjoint heldout states after8 fixed-rollout local updates. All arms share initialization and base-loss coefficient; augmented arms use extra verified suffix losses and extra compute. Error bars resample questions, not individual completions. Different panels use different question sets. StrictEM is not semantic/grounding verification. These pilots do not establish end-to-end efficacy, novelty, or a training-step trend.\n')
    print(OUT/'evidence_chain.png')


if __name__=='__main__':main()
