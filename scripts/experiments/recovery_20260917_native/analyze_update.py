"""Paired question-level summaries of fixed-state heldout generation, all arms."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'reports/recovery_20260917_native'


def main():
    rows={}
    for arm in ('zero','base','student_replay','teacher_repair'):
        p=OUT/('continue_student' if arm=='zero' else 'eval_'+arm);assert json.loads((p/'status.json').read_text())['state']=='complete'
        rr=[r for r in map(json.loads,(p/'continuations.jsonl').read_text().splitlines()) if r['split']=='eval']
        assert len(rr)==len({(r['qid'],r['replicate']) for r in rr})
        rows[arm]={(r['qid'],r['replicate']):r for r in rr}
        assert len(rr)==16 and all(r['split']=='eval' for r in rr)
    keys=set(rows['zero']);assert all(set(r)==keys for r in rows.values())
    for key in keys:
        assert all(rows[a][key]['initial_ids']==rows['zero'][key]['initial_ids'] for a in rows)
        assert all(rows[a][key]['seed']==rows['zero'][key]['seed'] for a in rows)
    equivalence=json.loads((OUT/'BASELINE_REUSE_EQUIVALENCE.json').read_text());assert equivalence['state']=='complete'
    qs=sorted({k[0] for k in keys});rng=np.random.default_rng(2026091717)
    per={a:{q:np.mean([r['em'] for k,r in rs.items() if k[0]==q]) for q in qs} for a,rs in rows.items()}
    arms={}
    for a,rr in rows.items():
        rs=list(rr.values());arms[a]=dict(correct=sum(r['em'] for r in rs),n=len(rs),calls=sum(r['search_calls'] for r in rs),generated_tokens=sum(r['generated_tokens'] for r in rs),seconds=sum(r['seconds'] for r in rs),stops=dict(collections.Counter(r['stop_reason'] for r in rs)))
    contrasts={}
    for a,b in [('base','zero'),('student_replay','base'),('teacher_repair','base'),('teacher_repair','student_replay')]:
        d=np.array([per[a][q]-per[b][q] for q in qs]);ci=np.quantile(d[rng.integers(0,len(d),size=(20000,len(d)))].mean(1),[.025,.975]).tolist()
        contrasts[a+' minus '+b]=dict(delta=float(d.mean()),question_ci95=ci,per_question=dict(zip(qs,d.tolist())))
    training={a:json.loads((OUT/('update_'+a)/'status.json').read_text()) for a in ('base','student_replay','teacher_repair')}
    result=dict(arms=arms,contrasts=contrasts,baseline_reused=True,baseline_equivalence=equivalence,training=training,scope='Exploratory8-question fixed-state continuation after8 fresh-optimizer local frozen-rollout updates; not end-to-end online training. Extra data acquisition/training compute unmatched. No novelty or grounding claim.')
    (OUT/'update_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    report=['# Local update pilot results','',result['scope'],'','|Arm|EM|Calls|Generated tokens|','|---|---:|---:|---:|']
    for a,r in arms.items():report.append(f"|{a}|{r['correct']}/{r['n']}|{r['calls']}|{r['generated_tokens']}|")
    report+=['','Original native baseline reused after CPU effective-inference-state equality check; no new zero-step generation.','']
    for name,r in contrasts.items():report.append(f"- {name}: {r['delta']*100:.2f}pp; exploratory question-bootstrap95%CI [{r['question_ci95'][0]*100:.2f},{r['question_ci95'][1]*100:.2f}]pp.")
    report+=['','Interpretation: positive point estimates only justify fresh independent confirmation; null estimates apply to this dose. Teacher versus student replay changes several properties of the data recipe simultaneously. Saved checkpoints are pilot artifacts, not replacement production models.']
    (OUT/'LOCAL_UPDATE_REPORT.md').write_text('\n'.join(report)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
