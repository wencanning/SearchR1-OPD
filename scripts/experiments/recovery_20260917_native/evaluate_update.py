"""Heldout continuation evaluation. Decoder copied from frozen recovery.py; no training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from probe import ROOT,GPU,STUDENT,TEACHER,save
INVALID_FEEDBACK='\nMy previous action is invalid. \tIf I want to search, I should put the query between <search> and </search>. \tIf I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['collect','continue'],required=True);p.add_argument('--model',choices=['student','teacher'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and os.environ.get('RECOVERY_GPU_GUARDED')=='1'
    assert args.phase=='continue' and args.model=='student'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteria,LogitsProcessor
    import requests
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
    source=ROOT/'reports/recovery_20260917_native/train_inputs.frozen.json';protocol=json.loads(source.read_text());cases=protocol['cases']
    args.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.checkpoint.resolve();assert path.is_relative_to(ROOT/'reports/recovery_20260917_native')
    weights={str(f):dict(size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in path.glob('*.safetensors')}
    manifest=dict(phase=args.phase,model=args.model,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint=str(path),weights=weights,retriever='http://127.0.0.1:8000/retrieve')
    if args.phase=='continue':
        sf=ROOT/'reports/recovery_20260917_native/collected/states.jsonl'
        assert json.loads(sf.with_name('status.json').read_text())['state']=='complete'
        manifest['states_sha256']=hashlib.sha256(sf.read_bytes()).hexdigest()
    mf=args.out/'manifest.frozen.json'
    if mf.exists():assert json.loads(mf.read_text())==manifest
    else:save(mf,manifest)
    session=requests.Session();session.trust_env=False
    cachefile=args.out/'retrieval_cache.json';cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True);st=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True);assert st.get_vocab()==tok.get_vocab()
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
    spec=importlib.util.spec_from_file_location('scorer',ROOT/'scripts/experiments/evidence_use/experiment.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
    eos=list(set([tok.eos_token_id,tok.convert_tokens_to_ids('<|endoftext|>')]))
    class Vocab(LogitsProcessor):
        def __call__(self,ids,scores):scores[:,len(tok):]=-float('inf');return scores
    class Stop(StoppingCriteria):
        def __init__(self,n):self.n=n
        def __call__(self,ids,scores,**kw):
            text=tok.decode(ids[0,self.n:],skip_special_tokens=False);return '</answer>' in text or '</search>' in text
    def generate(ids,budget):
        x=torch.tensor([ids],device='cuda')
        with torch.inference_mode():g=model.generate(x,attention_mask=torch.ones_like(x),do_sample=True,temperature=1.,top_p=1.,top_k=0,max_new_tokens=budget,use_cache=True,num_logits_to_keep=1,pad_token_id=tok.pad_token_id,eos_token_id=eos,logits_processor=[Vocab()],stopping_criteria=[Stop(len(ids))])[0,len(ids):].tolist()
        end=next((i for i,v in enumerate(g) if v in eos),len(g));g=g[:end];text=tok.decode(g,skip_special_tokens=False)
        match=re.search(r'<(search|answer)>(.*?)</\1>',text,re.S)
        valid=bool(match)
        return dict(ids=g,text=text,action=match[1] if valid else 'invalid',content=match[2].strip() if valid else None)
    def search(query):
        if query not in cache:
            r=session.post(manifest['retriever'],json=dict(queries=[query],topk=3,return_scores=True),timeout=120);r.raise_for_status();docs=r.json()['result'][0];assert len(docs)==3
            cache[query]=docs;save(cachefile,cache)
        docs=cache[query];body='\n'.join(f'Doc {i+1}(Title: '+d['document']['contents'].split('\n')[0].strip()+') '+'\n'.join(d['document']['contents'].split('\n')[1:]) for i,d in enumerate(docs))
        body_ids=tok.encode(body,add_special_tokens=False)
        obs=tok.encode('\n\n<information>',add_special_tokens=False)+body_ids[:512]+tok.encode('</information>\n\n',add_special_tokens=False)
        return obs
    result=args.out/'states.jsonl' if args.phase=='collect' else args.out/'continuations.jsonl'
    rows=[json.loads(l) for l in result.read_text().splitlines()] if result.exists() else []
    done={(r['qid'],r.get('replicate',0)) for r in rows};assert len(rows)==len(done)
    if args.phase=='continue':
        statefile=ROOT/'reports/recovery_20260917_native/collected/states.jsonl';collected=[json.loads(l) for l in statefile.read_text().splitlines()];stateby={r['qid']:r for r in collected}
        status=json.loads(statefile.with_name('status.json').read_text());assert status['state']=='complete'
        cases=[c for c in cases if stateby[c['qid']]['eligible'] and c['split']=='eval']
    reps=1 if args.phase=='collect' else protocol['replicates'];total=len(cases)*reps
    for c in cases:
        for rep in range(reps):
            if (c['qid'],rep) in done:continue
            seed=int(hashlib.sha256(f'2026091703|{args.phase}|{c["qid"]}|{rep}'.encode()).hexdigest()[:8],16);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);start=time.monotonic()
            ids=tok.apply_chat_template(c['prompt'],tokenize=True,add_generation_prompt=True) if args.phase=='collect' else list(stateby[c['qid']]['prefix_ids'])
            initial=list(ids);segments=[];calls=0;used=0;answer=None;stop=None;invalid_count=0
            while True:
                budget=min(256,768-used,4096-len(ids))
                if budget<=0:stop='token_or_context_limit';break
                g=generate(ids,budget);ids+=g['ids'];used+=len(g['ids']);segment=dict(g,loss_mask=[1]*len(g['ids']));segments.append(segment)
                if g['action']=='answer':answer=g['content'];stop='answer_closed';break
                if g['action']!='search':
                    if invalid_count>=1:stop='invalid_or_incomplete';break
                    invalid_count+=1;feedback_ids=tok.encode(INVALID_FEEDBACK,add_special_tokens=False);ids+=feedback_ids
                    segments.append(dict(action='invalid_feedback',ids=feedback_ids,text=INVALID_FEEDBACK,loss_mask=[0]*len(feedback_ids)))
                    continue
                if calls>=3:stop='search_limit';break
                obs=search(g['content']);calls+=1;ids+=obs;segments.append(dict(action='observation',ids=obs,text=tok.decode(obs),loss_mask=[0]*len(obs)))
                if args.phase=='collect':stop='state_collected';break
            row=dict(qid=c['qid'],split=c['split'],replicate=rep,seed=seed,model=args.model,initial_ids=initial,segments=segments,final_answer=answer,em=sc.exact(answer,c['gold']),stop_reason=stop,search_calls=calls,generated_tokens=used,seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated())
            if args.phase=='collect':row.update(eligible=stop=='state_collected' and len(ids)+256<=4096,prefix_ids=ids)
            with result.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            done.add((c['qid'],rep));save(args.out/'status.json',dict(state='running',completed=len(done),total=total));print(args.phase,args.model,len(done),total,c['qid'],row['em'],stop,flush=True)
    for f,m in weights.items():
        s=Path(f).stat();assert m==dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
    save(args.out/'status.json',dict(state='complete',completed=len(done),total=total))


if __name__=='__main__':main()
