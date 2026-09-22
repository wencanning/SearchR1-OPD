"""CPU audit of actual opening-tag tokenization and first-token OPD signals."""
import bisect
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/action-credit-recovery-20260914'
groups=defaultdict(list);files=[];examples=[]
for method in ['opd','sod']:
    for step in [75,85,95,100,105]:
        p=ROOT/f'verl_checkpoints/pure-{method}-05B-n1-20260914-1220-coef1/opd_diagnostics/step_{step:06d}.jsonl.gz'
        files.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        with gzip.open(p,'rt') as f:
            next(f)
            for line in f:
                r=json.loads(line);text=''.join(r['token_texts']);ends=[];pos=0
                for token in r['token_texts']:pos+=len(token);ends.append(pos)
                first_info=text.find('<information>')
                for m in re.finditer(r'<(think|search|answer)>',text):
                    a=bisect.bisect_right(ends,m.start());b=bisect.bisect_left(ends,m.end())+1
                    if not all(r['loss_mask'][i] for i in range(a,b)):continue
                    path=tuple(r['token_ids'][a:b]);post=first_info>=0 and m.start()>first_info
                    row=dict(index=r['index'],tag=m[1],post=post,tokens=r['token_texts'][a:b],ids=list(path),
                        first_teacher_logp=r['teacher_log_prob'][a],first_student_logp=r['student_log_prob'][a],
                        first_adv=r['opd_advantage'][a],first_weighted_adv=r['weighted_opd_advantage'][a],
                        tag_adv_sum=sum(r['opd_advantage'][a:b]))
                    groups[(method,step,m[1],post,r['token_ids'][a])].append(row)
                    if m[1]=='think' and post and len(examples)<12:examples.append(dict(method=method,step=step,**row))
results=[]
for key,rs in groups.items():
    method,step,tag,post,first_id=key
    results.append(dict(method=method,step=step,tag=tag,post=post,first_id=first_id,first_token=rs[0]['tokens'][0],n=len(rs),
        positive=sum(r['first_adv']>0 for r in rs),mean_first_adv=statistics.mean(r['first_adv'] for r in rs),
        mean_first_weighted_adv=statistics.mean(r['first_weighted_adv'] for r in rs),
        mean_first_teacher_logp=statistics.mean(r['first_teacher_logp'] for r in rs),
        mean_first_student_logp=statistics.mean(r['first_student_logp'] for r in rs)))
(OUT/'historical_protocol_signals.json').write_text(json.dumps(dict(results=results,files=files,examples=examples),ensure_ascii=False,indent=2))
for r in results:
    if r['tag']=='think' and r['post']:print(json.dumps(r,ensure_ascii=False),flush=True)
print('COMPLETE',flush=True)
