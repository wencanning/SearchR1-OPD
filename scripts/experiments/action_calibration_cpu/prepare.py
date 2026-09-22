"""Select disjoint calibration, test and update panels before model scoring."""
import collections
import json
from pathlib import Path
import re
from common import ROOT,OUT,M,NEUTRAL,CONDITIONS,sha,rank,save


def main():
    import pandas as pd
    from transformers import AutoTokenizer
    assert not (OUT/'candidates.json').exists(),'Refusing to redraw a selected panel'
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    test_path=ROOT/'data/nq_hotpotqa_train/test.parquet'
    train_path=ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet'
    val_path=ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet'
    test=pd.read_parquet(test_path);excluded=set(map(M.norm,pd.read_parquet(train_path).question))|set(map(M.norm,pd.read_parquet(val_path).question))
    prior=[ROOT/'reports/evidence_gap_gpu23_20260915/panels.json',ROOT/'reports/evidence_use_gpu5_20260915/panel.json',
           ROOT/'reports/evidence_stopping_gpu5_20260915/candidates.json',ROOT/'reports/evidence_stopping_gpu5_20260915/natural_panel.json']
    for p in prior:
        raw=M.read(p);groups=raw.values() if isinstance(raw,dict) else [raw]
        for group in groups:excluded.update(M.norm(q['question']) for q in group)
    candidates=[];seen=set();counts=collections.Counter()
    for q in test[test.data_source=='hotpotqa'].to_dict('records'):
        normalized=M.norm(q['question']);counts['all_hotpot']+=1
        if normalized in excluded or normalized in seen:continue
        seen.add(normalized);metadata=q['metadata']
        if metadata['type']!='bridge':continue
        titles=list(dict.fromkeys(metadata['supporting_facts']['title']))
        ctx=dict(zip(metadata['context']['title'],metadata['context']['sentences']))
        if len(titles)!=2 or not all(t in ctx for t in titles):continue
        sf=collections.defaultdict(list)
        for t,i in zip(metadata['supporting_facts']['title'],metadata['supporting_facts']['sent_id']):
            if 0<=i<len(ctx[t]):sf[t].append(ctx[t][i])
        choices=[]
        for a,b in (titles,titles[::-1]):
            bridge=M.base_title(b);gold=list(q['golden_answers'])
            if not M.mention(M.base_title(a),q['question']) or M.mention(bridge,q['question']):continue
            if not M.mention(bridge,' '.join(sf[a])):continue
            if not any(M.mention(g,' '.join(sf[b])) for g in gold):continue
            if any(M.mention(g,''.join(ctx[a])) for g in gold):continue
            d1,d2=M.doc(a,ctx[a]),M.doc(b,ctx[b])
            if max(len(tok.encode(d['text'])) for d in [d1,d2])>384:continue
            noise=[M.doc(t,ss) for t,ss in ctx.items() if t not in titles and not M.mention(bridge,t+' '+''.join(ss))
                and not any(M.mention(g,t+' '+''.join(ss)) for g in gold)]
            noise=[d for d in noise if len(tok.encode(d['text']))<=384]
            noise.sort(key=lambda d:(abs(len(tok.encode(d['text']))-len(tok.encode(d1['text']))),d['title']))
            if len(noise)<2:continue
            choices.append(dict(qid=str(q['id']),question=q['question'],gold=gold,prompt=[dict(p) for p in q['prompt']],
                first=d1,second=d2,bridge=bridge,distractors=noise[:2],first_support=list(sf[a]),second_support=list(sf[b])))
        if len(choices)==1:candidates.append(choices[0])
    candidates.sort(key=lambda q:rank(q['qid'],'action-calibration-20260916'))
    assert len(candidates)>=72,len(candidates)
    # Input review may exclude ambiguous items; split assignment happens only after review.
    panel=candidates[:72]
    save(OUT/'candidates.json',panel)
    save(OUT/'selection.json',dict(eligible=len(candidates),review_candidates=72,split_rule='First 56 input-approved hash-ranked candidates: calibration16, test32, update8',
         source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [test_path,train_path,val_path]+prior},excludes_training_validation_and_all_prior_panels=True))
    save(OUT/'review.json',[dict(qid=q['qid'],include=None,reason='') for q in panel])
    lines=['# 输入审查，模型评分之前','']
    for q in panel:
        lines += [f"{q['qid']} | {q['question']} | gold={q['gold']}",
                  'First: '+' '.join(q['first_support']),'Second: '+' '.join(q['second_support']),'']
    (OUT/'INPUT_REVIEW.md').write_text('\n'.join(lines))
    print('PREPARED_INPUT_REVIEW',len(panel),'eligible',len(candidates),flush=True)


def freeze():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    review={q['qid']:q for q in M.read(OUT/'review.json')}
    assert all(isinstance(q['include'],bool) and q['reason'] for q in review.values())
    selected=[q for q in M.read(OUT/'candidates.json') if review[q['qid']]['include']][:56]
    assert len(selected)==56
    groups={'calibration':selected[:16],'test':selected[16:48],'update':selected[48:]}
    cases=[]
    for split,qs in groups.items():
        for q in qs:
            for condition in CONDITIONS:
                ids,_=M.make_prefix(tok,q,condition)
                evidence_text=tok.decode(ids,skip_special_tokens=False)
                a,b=list(re.finditer(r'<information>(.*?)</information>',evidence_text,re.S))[-1].span(1)
                lengths=[0]+[len(tok.decode(ids[:i],skip_special_tokens=False)) for i in range(1,len(ids)+1)]
                em=[int(y>x and x>=a and y<=b) for x,y in zip(lengths,lengths[1:])]
                history=tok.encode(NEUTRAL,add_special_tokens=False);ids+=history;em += [0]*len(history)
                assert len(ids)+96<1536
                cases.append(dict(qid=q['qid'],split=split,condition=condition,ids=ids,evidence_mask=em,gold=q['gold']))
    checkpoints=[]
    for step in [50,100]:
        for method,folder in [('opd','opd-grpo-0.5B'),('er','eropd-grpo-05B-test-alpha2')]:
            checkpoints.append(dict(label=f'{method}-step{step}',method=method,step=step,path=f'verl_checkpoints/{folder}/actor/global_step_{step}'))
    sources=[OUT/'candidates.json',OUT/'review.json',OUT/'selection.json',Path(__file__),ROOT/'scripts/experiments/action_calibration_cpu/common.py',ROOT/'scripts/experiments/evidence_use/experiment.py']
    inputs=dict(cases=cases,split_qids={k:[q['qid'] for q in v] for k,v in groups.items()},checkpoints=checkpoints,
        tags={a:tok.encode('<'+a+'>',add_special_tokens=False) for a in ['answer','search']},vocab_size=len(tok),
        input_review_passed=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        device='cpu',dtype='float32',history='Same neutral synthetic pre-action thought in all conditions and students',
        bias_rule='One b per predeclared checkpoint; calibrate mean sigmoid(OPD log-odds+b) to mean ER conditional answer probability using calibration only',
        primary_step=100,secondary_step=50,probability_equivalence_tolerance=.10,
        update=dict(start_checkpoint='verl_checkpoints/opd-grpo-0.5B/actor/global_step_50',qids=[q['qid'] for q in groups['update']],
            conditions=['first_hop','both_hops'],samples_per_state=2,max_new_tokens=96,sampling_temperature=1.0,seed=2026091602,
            alpha=1.5,lambda_distill=.01,common_grpo_advantage=0.0,
            isolation='Local distillation-component diagnostic; no fabricated terminal reward for truncated search, not full OPD+GRPO historical replay',
            optimizer='fresh AdamW lr1e-6 betas(.9,.999) eps1e-8 weight_decay0',grad_clip=1.0,
            variants=['opd','full_er','action_only_er','body_only_er']))
    save(OUT/'inputs.frozen.json',inputs)
    print('FROZEN',inputs['split_qids'],flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='freeze':freeze()
    else:main()
