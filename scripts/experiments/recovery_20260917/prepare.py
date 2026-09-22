"""Freeze input-only four-cell evidence/history diagnostic; no outcome selection."""
import hashlib
import json
import os
from pathlib import Path
import re
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/recovery_20260917'


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source=ROOT/'reports/er_next_20260916/fork_states.frozen.json'
    states=json.loads(source.read_text())['states']
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    data=ROOT/'data/nq_hotpotqa_train/test.parquet'
    by={r['question']:r for r in pq.read_table(data,columns=['id','question','metadata','data_source']).to_pylist() if r['data_source']=='hotpotqa'}
    cases=[];coverage=[]
    for s in states:
        row=by[s['question']];m=row['metadata'];ctx=dict(zip(m['context']['title'],m['context']['sentences']))
        sf=list(zip(m['supporting_facts']['title'],m['supporting_facts']['sent_id']))
        if not all(t in ctx and 0<=i<len(ctx[t]) for t,i in sf):
            coverage.append(dict(qid=s['qid'],reason='missing_annotated_support'));continue
        titles=list(dict.fromkeys(t for t,i in sf))
        docs=[f'Doc {j+1}(Title: {t}) '+' '.join(ctx[t][i] for tt,i in sf if tt==t) for j,t in enumerate(titles)]
        support='\n'.join(docs)
        if len(tok.encode(support,add_special_tokens=False))>512:
            coverage.append(dict(qid=s['qid'],reason='annotated_support_over_512'));continue
        text=tok.decode(s['prefix_ids'],skip_special_tokens=False)
        assistant_start=text.rfind('<|im_start|>assistant\n')
        assert assistant_start>=0
        matches=[v for v in re.finditer(r'<information>(.*?)</information>',text,re.S) if v.start()>assistant_start]
        assert len(matches)==1
        obs=matches[0]
        tail=text[obs.end():]
        assert re.fullmatch(r'\s*<think>.*?</think>\s*',tail,re.S),repr(tail)
        variants={}
        for evidence in ('actual','support'):
            prefix=text if evidence=='actual' else text[:obs.start(1)]+support+text[obs.end(1):]
            close=prefix.rfind('</information>')+len('</information>')
            for history in ('native','neutral'):
                value=prefix if history=='native' else prefix[:close]+'\n\n<think></think>\n\n'
                ids=s['prefix_ids'] if evidence=='actual' and history=='native' else tok.encode(value,add_special_tokens=False)
                assert len(ids)+128<=4096
                variants[evidence+'_'+history]=dict(prefix_ids=ids,prefix_text=value,tokens=len(ids))
        cases.append(dict(qid=s['qid'],question=s['question'],gold=s['gold'],support_facts=[dict(title=t,sentence_id=i,text=ctx[t][i]) for t,i in sf],support_tokens=len(tok.encode(support)),variants=variants))
        coverage.append(dict(qid=s['qid'],reason='included'))
    payload=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_data=str(data),source_data_size=data.stat().st_size,source_data_mtime_ns=data.stat().st_mtime_ns,
        prepare_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,coverage=coverage,replicates=2,max_new_tokens=96,temperature=1.,top_p=1.,top_k=0,
        qualification='Previously inspected 64-question exploratory panel. Input coverage selection only; no outcome selection. Supporting-fact snippets are privileged diagnosis, not deployed retrieval. Neutral removes only latest post-observation think; changes length too. No training on these cases.')
    OUT.mkdir(exist_ok=True,parents=True);path=OUT/'probe_inputs.frozen.json'
    if path.exists():assert json.loads(path.read_text())==payload
    else:path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(included=len(cases),population=len(states),planned_generations=len(cases)*4*2*2)))


if __name__=='__main__':main()
