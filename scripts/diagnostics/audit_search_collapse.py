"""Read-only post-hoc collapse audit; separate action-keyword signals from query text."""
import bisect
import csv
import gzip
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/independent_history_20260914'
TAG='20260914-1220-coef1'


def main():
    rows=[];actions=[];files=[]
    # Different fixed endpoints: these are already complete dumps when the user reported collapse.
    for method,stop in [('opd',106),('sod',110)]:
        for step in range(90,stop+1):
            path=ROOT/f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
            with gzip.open(path,'rt') as f:
                header=json.loads(next(f));rs=[json.loads(l) for l in f]
            assert len(rs)==128 and header['metrics']['opd/grpo_advantage']==0
            files.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            groups=defaultdict(list);no_answer=0;five_search=0;repeat=0
            for r in rs:
                text=''.join(r['token_texts']);ends=[];pos=0
                for token in r['token_texts']:pos+=len(token);ends.append(pos)
                no_answer+=not bool(re.search(r'<answer>.*?</answer>',text,re.S))
                searches=re.findall(r'<search>(.*?)</search>',text,re.S)
                five_search+=len(searches)>=5
                normalized=[' '.join(q.lower().split()) for q in searches]
                repeat+=len(normalized)>len(set(normalized))
                seen_search=0
                for match in re.finditer(r'<(search|answer|think)>',text):
                    kind=match[1];start=match.start(1);end=match.end(1)
                    left=bisect.bisect_right(ends,start);right=bisect.bisect_left(ends,end)+1
                    ids=list(range(left,right))
                    if not ids or not all(r['loss_mask'][i] for i in ids):continue
                    gap=sum(r['opd_advantage'][i] for i in ids)
                    weighted=sum(r['weighted_opd_advantage'][i] for i in ids)
                    group=kind if kind!='search' else ('search_first' if seen_search==0 else 'search_later')
                    if kind=='search':seen_search+=1
                    event=dict(method=method,step=step,index=r['index'],group=group,keyword=kind,
                        tokens=[r['token_texts'][i] for i in ids],advantage_sum=gap,weighted_sum=weighted,
                        teacher_logp_sum=sum(r['teacher_log_prob'][i] for i in ids),
                        student_logp_sum=sum(r['student_log_prob'][i] for i in ids))
                    actions.append(event);groups[group].append(event)
            row=dict(method=method,step=step,n=128,no_answer=no_answer,five_search_tags=five_search,repeated_exact_query=repeat)
            for group in ['search_first','search_later','answer','think']:
                events=groups[group];row[group+'_n']=len(events)
                row[group+'_positive_fraction']=statistics.mean(e['advantage_sum']>0 for e in events) if events else None
                row[group+'_mean_adv']=statistics.mean(e['advantage_sum'] for e in events) if events else None
                row[group+'_mean_weighted_adv']=statistics.mean(e['weighted_sum'] for e in events) if events else None
            rows.append(row)
    metrics=[]
    val=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet')
    counts=val['data_source'].value_counts().to_dict()
    for method in ['opd','sod']:
        path=ROOT/f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/train.log'
        for line in path.open():
            m=re.search(r'step:(\d+) - ',line)
            if not m or not 70<=int(m[1])<=110:continue
            parsed={k:float(v) for k,v in re.findall(r'([\w/]+):(-?[\d.]+)',line)}
            r=dict(method=method,step=int(m[1]),**{k:parsed.get(k) for k in
                ['critic/score/mean','env/finish_ratio','env/number_of_valid_search','opd/divergence','actor/grad_norm','opd/student_entropy']})
            if 'val/test_score/Avg' in parsed:
                correct={k:round(n*parsed['val/test_score/'+k]) for k,n in counts.items()}
                assert all(abs(correct[k]/n-parsed['val/test_score/'+k])<=.000501 for k,n in counts.items())
                r['val_correct']=sum(correct.values());r['val_n']=512
            metrics.append(r)
    (OUT/'collapse_audit.json').write_text(json.dumps(dict(rows=rows,metrics=metrics,files=files),ensure_ascii=False,indent=2))
    (OUT/'action_keyword_signals.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in actions))
    with (OUT/'collapse_by_step.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps([r for r in rows if r['step'] in [90,95,98,99,100,104,106,108,110]],ensure_ascii=False,indent=2))
    print('VALIDATION',[(r['method'],r['step'],r['val_correct']) for r in metrics if 'val_correct' in r])


if __name__=='__main__':main()
