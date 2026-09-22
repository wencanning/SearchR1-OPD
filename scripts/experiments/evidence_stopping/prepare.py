"""Select fresh inputs before any follow-up model outputs are available."""
import collections
import importlib.util
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/evidence_stopping_gpu5_20260915'
spec = importlib.util.spec_from_file_location('previous_pilot', ROOT/'scripts/experiments/evidence_use/experiment.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def main():
    import pandas as pd
    from transformers import AutoTokenizer
    if (OUT/'candidates.json').exists(): raise RuntimeError('Do not redraw a selected panel')
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    test=pd.read_parquet(ROOT/'data/nq_hotpotqa_train/test.parquet')
    train=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    old=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet')
    excluded=set(map(M.norm,train.question))|set(map(M.norm,old.question))
    for group in M.read(ROOT/'reports/evidence_gap_gpu23_20260915/panels.json').values():
        excluded.update(M.norm(q['question']) for q in group)
    excluded.update(M.norm(q['question']) for q in M.read(ROOT/'reports/evidence_use_gpu5_20260915/panel.json'))
    pool=[];seen=set()
    for q in test[test.data_source=='hotpotqa'].to_dict('records'):
        nq=M.norm(q['question'])
        if nq not in excluded and nq not in seen: pool.append(q);seen.add(nq)
    eligible=[]
    for q in pool:
        m=q['metadata']
        if m['type']!='bridge':continue
        titles=list(dict.fromkeys(m['supporting_facts']['title']))
        ctx=dict(zip(m['context']['title'],m['context']['sentences']))
        if len(titles)!=2 or not all(t in ctx for t in titles):continue
        sf=collections.defaultdict(list)
        for t,i in zip(m['supporting_facts']['title'],m['supporting_facts']['sent_id']):
            if 0<=i<len(ctx[t]):sf[t].append(ctx[t][i])
        choices=[]
        for a,b in (titles,titles[::-1]):
            bridge=M.base_title(b); gold=list(q['golden_answers'])
            if not M.mention(M.base_title(a),q['question']) or M.mention(bridge,q['question']):continue
            if not M.mention(bridge,' '.join(sf[a])):continue
            if not any(M.mention(g,' '.join(sf[b])) for g in gold):continue
            if any(M.mention(g,''.join(ctx[a])) for g in gold):continue
            d1,d2=M.doc(a,ctx[a]),M.doc(b,ctx[b])
            if max(len(tok.encode(d['text'])) for d in (d1,d2))>512:continue
            noise=[M.doc(t,ss) for t,ss in ctx.items() if t not in titles
                and not M.mention(bridge,t+' '+''.join(ss))
                and not any(M.mention(g,t+' '+''.join(ss)) for g in gold)]
            noise=[d for d in noise if len(tok.encode(d['text']))<=512]
            noise.sort(key=lambda d:(abs(len(tok.encode(d['text']))-len(tok.encode(d1['text']))),d['title']))
            if len(noise)<2:continue
            choices.append(dict(qid=str(q['id']),question=q['question'],gold=gold,
                prompt=[dict(p) for p in q['prompt']],first=d1,second=d2,bridge=bridge,
                distractors=noise[:2],first_support=list(sf[a]),second_support=list(sf[b])))
        if len(choices)==1:eligible.append(choices[0])
    eligible.sort(key=lambda q:q['qid']); random.Random(2026091506).shuffle(eligible)
    panel=eligible[:48]
    assert len(panel)==48
    M.save(OUT/'candidates.json',panel)
    candidate_questions={M.norm(q['question']) for q in panel}
    natural=[q for q in pool if M.norm(q['question']) not in candidate_questions]
    natural.sort(key=lambda q:q['id']);random.Random(2026091507).shuffle(natural)
    natural=[dict(qid=str(q['id']),question=q['question'],gold=list(q['golden_answers']),
        prompt=[dict(p) for p in q['prompt']],type=q['metadata']['type']) for q in natural[:64]]
    M.save(OUT/'natural_panel.json',natural)
    M.save(OUT/'selection.json',dict(eligible=len(eligible),fresh_hotpot_pool=len(pool),
        controlled_seed=2026091506,natural_seed=2026091507,
        candidates_sha256=M.digest(panel),natural_sha256=M.digest(natural),
        natural_types=dict(collections.Counter(q['type'] for q in natural))))
    M.save(OUT/'review.json',[dict(qid=q['qid'],include=None,reason='',aliases=q['gold']) for q in panel])
    lines=['# Fresh input-only review; before model outputs\n']
    for q in panel:
        lines += [f"## {q['qid']}: {q['question']}",f"Reference: {q['gold']}",
            f"First: {q['first']['title']} -> Second: {q['second']['title']}",
            'First support: '+' '.join(q['first_support']),
            'Second support: '+' '.join(q['second_support'])+'\n']
    (OUT/'INPUT_REVIEW.md').write_text('\n\n'.join(lines))
    print(json.dumps(M.read(OUT/'selection.json'),indent=2))


if __name__=='__main__': main()
