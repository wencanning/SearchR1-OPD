"""Read append-only partial or final diagnostic outputs; no GPU use."""
import json
from collections import defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parents[2]/'reports/search_loop_20260914'

def read(name):
    if not (OUT/name).exists():return []
    rows=[]
    for line in (OUT/name).read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break  # concurrently appended final line
    return rows

def main():
    groups=defaultdict(list)
    for r in read('generations_v2.jsonl'):
        groups[(r['model'],r['source_method'],r['kind'],r['state'],r['seed'])].append(r)
    print('model,source,kind,state,seed,n,repeat,search,answer,incomplete')
    for key,rs in groups.items():
        print(','.join(map(str,(*key,len(rs),sum(r['repeat'] is True for r in rs),
            sum(r['action']=='search' for r in rs),sum(r['action']=='answer' for r in rs),sum(r['action']=='incomplete' for r in rs)))))
    scores=read('scores_v2.jsonl');checked=[r for r in scores if 'numerical_pass' in r]
    print('SCORES',len(scores),'TEACHER_PARITY',len(checked),'FAIL',sum(not r['numerical_pass'] for r in checked))
    gpu={(r['key'],r['state']):r for r in scores if r['model']=='teacher'}
    for r in read('cpu_eager.jsonl'):
        match=gpu.get((r['key'],r['state']))
        if match and 'fresh_logps' in match:
            print('CPU_GPU_FP32',r['key'],r['state'],max(abs(a-b) for a,b in zip(r['logps'],match['fresh_logps'])))
    for r in read('updates.jsonl'):print('UPDATE',r['method'],r['variant'],r['status'],r.get('grad_norm'))
    print('MILESTONES',read('milestones.jsonl'))

if __name__=='__main__':main()
