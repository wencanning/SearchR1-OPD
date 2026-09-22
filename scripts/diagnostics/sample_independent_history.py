"""Freeze a new, outcome-stratified sample; keep teacher signals out of review cards."""
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from audit_pure_opd_mainline import features

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/independent_history_20260914'
TAG='20260914-1220-coef1'
SEED='independent-history-20260914-v1'


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def old_indices():
    paths=['paired_coef1_first20/review_sample.json','pure_opd_mainline_20260914/review_sample.json',
           'offline_er_cpu/manifest.json','offline_er_cpu_v2/manifest.json','offline_er_cpu_v3/manifest.json',
           'history_intervention_cpu_20260914/manifest.json','claim_intervention_cpu_20260914/manifest.json',
           'evidence_update_cpu_20260914/teacher_cases_first12.json']
    seen=set()
    def walk(x):
        if isinstance(x,dict):
            if 'index' in x and str(x['index']).isdigit():seen.add(int(x['index']))
            for k in ['key','case_id']:
                if re.fullmatch(r'(opd|sod):\d+:\d+',str(x.get(k,''))):seen.add(int(x[k].split(':')[-1]))
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for name in paths:
        p=ROOT/'reports'/name
        if p.exists():walk(json.loads(p.read_text()))
    return sorted(seen)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'sample_manifest.json').exists(), 'Frozen sample already exists; never overwrite'
    frame=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    lookup={(r['data_source'],int(r['extra_info']['index'])):r for r in frame.to_dict('records')}
    excluded=old_indices(); population=[]; hashes={}; selected=[]; pool_counts={}
    for method in ['opd','sod']:
        candidates=[]
        for step in range(91,101):
            path=ROOT/f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
            relative=str(path.relative_to(ROOT)); hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path,'rt') as f:
                header=json.loads(next(f)); rows=[json.loads(l) for l in f]
            assert header['batch_size']==header['selected_sequences']==len(rows)==128
            for r in rows:
                index=int(r['index']); source=lookup[(r['data_source'],index)]
                gold=list(source['reward_model']['ground_truth']['target'])
                text=''.join(r['token_texts']); feat=features(text,gold)
                assert feat['score']==r['sequence_score']
                key=f'{method}:{step}:{index}'
                ids=[i for i,(s,m) in enumerate(zip(r['segments'],r['loss_mask'])) if s=='answer' and m]
                # Descriptive signals are saved separately, not printed on semantic review cards.
                pop=dict(key=key,method=method,step=step,index=index,source_file=relative,**feat,
                         answer_token_n=len(ids),answer_adv_sum=sum(r['opd_advantage'][i] for i in ids),
                         answer_weighted_adv_sum=sum(r['weighted_opd_advantage'][i] for i in ids))
                population.append(pop)
                if index not in excluded:
                    candidates.append(dict(**pop,question=source['question'],gold=gold,response=text))
        # Deduplicate questions BEFORE stratified sampling, with a score-independent hash choice.
        unique={}
        for r in sorted(candidates,key=lambda r:sha(SEED+'|dedup|'+r['key'])):
            unique.setdefault(r['index'],r)
        pool_counts[method]=dict(all_window=1280,prior_question_excluded=1280-len(candidates),
            duplicate_rows_removed=len(candidates)-len(unique),
            available_after_prior_exclusion=len(candidates),unique_questions=len(unique),
            outcome_counts=dict(Counter(r['score'] for r in unique.values())))
        for score,n in [(0,20),(1,5)]:
            pool=sorted([r for r in unique.values() if r['score']==score],key=lambda r:sha(SEED+'|sample|'+r['key']))
            assert len(pool)>=n
            selected.extend(pool[:n])
    selected.sort(key=lambda r:sha(SEED+'|blind|'+r['key']))
    cards=[]
    for i,r in enumerate(selected,1):
        r['review_id']=f'T{i:03d}'
        cards.append({k:r[k] for k in ['review_id','question','gold','response']})
    manifest=dict(seed=SEED,window=[91,100],excluded_prior_indices=excluded,source_hashes=hashes,
        pool_counts=pool_counts,cases=selected)
    (OUT/'sample_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (OUT/'review_cards.json').write_text(json.dumps(cards,ensure_ascii=False,indent=2))
    (OUT/'population_features.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in population))
    print(json.dumps(dict(n_population=len(population),n_review=len(cards),excluded_prior_questions=len(excluded),
        manifest_sha256=hashlib.sha256((OUT/'sample_manifest.json').read_bytes()).hexdigest(),
        pool_counts=pool_counts),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
