"""Input-only, question-disjoint train/eval pilot selection; no gold prompting."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[3]


def norm(s):return ' '.join(s.lower().split())


def main():
    import pyarrow.parquet as pq
    train=pq.read_table(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet').to_pylist()
    test=pq.read_table(ROOT/'data/nq_hotpotqa_train/test.parquet').to_pylist()
    trainq={norm(r['question']) for r in train}
    excluded=set()
    sources=['reports/er_next_20260916/fork_states.frozen.json','reports/evidence_use_gpu5_20260915/panel.json','reports/evidence_stopping_gpu5_20260915/candidates.json','reports/evidence_gap_gpu23_20260915/panels.json']
    def walk(x):
        if isinstance(x,dict):
            if isinstance(x.get('question'),str):excluded.add(norm(x['question']))
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for source in sources:
        if (ROOT/source).exists():walk(json.loads((ROOT/source).read_text()))
    diagnostic=pq.read_table(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet',columns=['question']).to_pylist()
    excluded.update(norm(r['question']) for r in diagnostic)
    cases=[]
    for split,rows,limit in [('train',train,16),('eval',test,8)]:
        eligible={norm(r['question']):r for r in rows if r['data_source']=='hotpotqa' and norm(r['question']) not in excluded and (split=='train' or norm(r['question']) not in trainq)}
        order=sorted(eligible,key=lambda q:hashlib.sha256(('recovery20260917|'+split+'|'+q).encode()).hexdigest())
        assert len(order)>=limit
        for q in order[:limit]:
            r=eligible[q];cases.append(dict(qid=str(r['id']),split=split,question=r['question'],gold=list(r['golden_answers']),prompt=[dict(p) for p in r['prompt']]))
    assert len({norm(c['question']) for c in cases})==len(cases)
    payload=dict(cases=cases,replicates=2,max_segment=256,max_generated=768,max_new_searches=3,max_context=4096,topk=3,observation_tokens=512,seed=2026091703,
      prepare_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),excluded_sources=sources,
      qualification='Input-only 16 HotpotQA training and8 question-disjoint heldout pilot questions; existing known panels excluded. Generator never sees gold. Exploratory tiny pilot, no full benchmark claim.')
    path=ROOT/'reports/recovery_20260917/train_inputs.frozen.json'
    if path.exists():assert json.loads(path.read_text())==payload
    else:path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print('TRAIN_INPUTS_FROZEN',len(cases))


if __name__=='__main__':main()
