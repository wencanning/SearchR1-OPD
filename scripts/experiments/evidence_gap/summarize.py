"""Tables from real JSONL only. Reference EM is explicitly distinct from semantic correctness."""
import csv
import math
from pathlib import Path
from common import METHODS, TARGETS, read, rows, save


def interval(values, repetitions=2000, seed=20260915):
    import numpy as np
    if not values: return dict(n=0,mean=None,low=None,high=None)
    x=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(repetitions,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)))


def sigmoid(x):
    return 1/(1+math.exp(-max(-700,min(700,x))))


def write_csv(path,records):
    if not records: return
    keys=list(dict.fromkeys(k for row in records for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(records)


def summarize(c,label):
    folder=Path(c['output'])/label
    scores=rows(folder/'scores-0.jsonl')+rows(folder/'scores-1.jsonl')
    candidates=rows(folder/'candidates-0.jsonl')+rows(folder/'candidates-1.jsonl')
    if not candidates or {r['key'] for r in scores}!={r['key'] for r in candidates}:
        raise ValueError('Incomplete diagnostic scoring; no final figure tables emitted')
    report=dict(label=label,data_kind='REAL_EXPERIMENT',teacher_dtype=c['teacher_dtype'],alpha=c['alpha'],A=[],B=[],C=[],generation=[],denominators=[])
    for step in sorted({r['step'] for r in candidates}):
        allstep=[r for r in scores if r['step']==step]; valid=[r for r in allstep if 'excluded' not in r]
        n=len(valid)
        from collections import Counter
        original=[r for r in candidates if r['step']==step]
        stops=Counter(r.get('stop_reason','unrecorded') for r in original)
        technical=Counter(r['technical_exclusion'] for r in original if r.get('technical_exclusion'))
        reasons=Counter(r['excluded'] for r in allstep if 'excluded' in r)
        report['denominators'].append(dict(step=step,sampled=len(original),eligible=n,
            complete_answers=sum(r.get('answer') is not None for r in original),
            annotation_sufficient=sum(bool(r.get('annotation',{}).get('sufficient')) for r in allstep),
            annotation_insufficient=sum(r.get('annotation',{}).get('sufficient') is False for r in allstep),
            **{f'technical_{k}':v for k,v in technical.items()},
            **{f'excluded_{k}':v for k,v in reasons.items()},**{f'stopped_{k}':v for k,v in stops.items()}))
        for epsilon in (0,.05,.1):
            # Epsilon > 0 is a boundary sensitivity; explicit uncertain-band count retained.
            good=[r for r in valid if r['margins']['observed']>epsilon]
            gap=[r for r in valid if r['margins']['observed'] < -epsilon and r['margins']['observed']-r['margins']['hidden']>epsilon]
            nonhelp=[r for r in valid if r['margins']['observed'] <= 0 and r['margins']['observed']-r['margins']['hidden']<=0]
            classified={r['key'] for r in good+gap+nonhelp}
            for name,subset in [('Teacher favors correct',good),('Evidence helps, still wrong',gap),('No helpful update',nonhelp)]:
                keys={r['key'] for r in subset}
                report['A'].append(dict(step=step,epsilon=epsilon,metric=name,sampled=len(allstep),eligible=n,
                    excluded=len(allstep)-n,boundary_unclassified=n-len(classified),
                    **interval([int(r['key'] in keys) for r in valid],c['bootstrap_samples'])))
        cohort=[r for r in valid if r['margins']['hidden']<r['margins']['observed']<0]
        for target in TARGETS:
            report['B'].append(dict(step=step,target=target,metric='Pairwise correct-answer support',
                **interval([sigmoid(r['margins'][target]) for r in cohort],c['bootstrap_samples'])))
            for group,rr in [('helps_still_wrong',cohort),('all_sufficient',valid)]:
                report['generation'].append(dict(step=step,target=target,group=group,
                    metric='Generated answer: reviewed-alias EM',
                    **interval([int(r['generations'][target]['reviewed_alias_em']) for r in rr],c['bootstrap_samples'])))
                report['generation'].append(dict(step=step,target=target,group=group,metric='Net generated correction vs observed',
                    **interval([int(r['generations'][target]['reviewed_alias_em'])-int(r['generations']['observed']['reviewed_alias_em']) for r in rr],c['bootstrap_samples'])))
                report['generation'].append(dict(step=step,target=target,group=group,metric='Net candidate flip vs observed',
                    **interval([int(r['margins'][target]>0)-int(r['margins']['observed']>0) for r in rr],c['bootstrap_samples'])))
                for metric,start,end in [('Generated correction: wrong to correct',False,True),
                        ('Generated harm: correct to wrong',True,False),
                        ('Generated retention: correct to correct',True,True)]:
                    subset=[r for r in rr if bool(r['generations']['observed']['reviewed_alias_em'])==start]
                    vals=[int(bool(r['generations'][target]['reviewed_alias_em'])==end) for r in subset]
                    report['generation'].append(dict(step=step,target=target,group=group,metric=metric,
                        transitions=sum(vals),denominator=len(subset),**interval(vals,c['bootstrap_samples'])))
    # C requires all predeclared methods/seeds/checkpoints/questions. Do not cherry-pick completed runs.
    panels=read(Path(c['output'])/'panels.json'); qids=[r['qid'] for r in panels['evaluation']]
    data={}; missing=[]
    for m in METHODS:
        for seed in c['seeds']:
            d=Path(c['output'])/f'evaluation-{m}-seed{seed}'
            rr=rows(d/'rollouts-0.jsonl')+rows(d/'rollouts-1.jsonl')
            table={(r['step'],r['qid']):r for r in rr}; data[m,seed]=table
            if not (d/'evaluate.complete.json').exists() or any((s,q) not in table for s in c['steps'] for q in qids):
                missing.append(f'{m}-seed{seed}')
    report['C_pending']=missing
    if not missing:
        import numpy as np
        base=data['opd',c['seeds'][0]]
        initial={q:base[0,q]['reference_em'] for q in qids}
        for table in data.values():
            if any(table[0,q]['answer']!=base[0,q]['answer'] for q in qids):
                raise ValueError('Initial greedy generations differ across runs; investigate retriever/evaluation drift')
        for m in METHODS:
            for step in c['steps']:
                for name,panel,field in [('Correction of initial errors',[q for q in qids if not initial[q]],'reference_em'),
                      ('Retention of initial correct answers',[q for q in qids if initial[q]],'reference_em'),
                      ('Overall exact match',qids,'reference_em'),('Overall F1',qids,'reference_f1')]:
                    if not panel: continue
                    x=np.array([[float(data[m,s][step,q][field]) for q in panel] for s in c['seeds']])
                    rng=np.random.default_rng(c['split_seed']); means=[]
                    for _ in range(c['bootstrap_samples']):
                        si=rng.integers(0,len(c['seeds']),len(c['seeds'])); qi=rng.integers(0,len(panel),len(panel))
                        means.append(x[si][:,qi].mean())
                    report['C'].append(dict(method=m,step=step,metric=name,n_questions=len(panel),n_seeds=len(c['seeds']),
                        mean=float(x.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),
                        seed_means=x.mean(1).tolist()))
    save(folder/'summary.json',report)
    for name in ('A','B','C','generation','denominators'): write_csv(folder/f'{name}.csv',report[name])
    # New third-answer errors remain visible for blind semantic adjudication; never silently expand primary aliases.
    adjudication=[]
    for r in scores:
        if 'excluded' in r: continue
        for target,g in r['generations'].items():
            adjudication.append(dict(key=r['key'],target=target,answer=g['answer'],stop_reason=g['stop_reason'],
                 reference_em=g['reference_em'],reviewed_alias_em=g['reviewed_alias_em'],semantic_correct=None))
    save(folder/'generation_semantic_review.json',adjudication)
    print(f'Real tables written: {folder}/summary.json; C pending: {missing}')
