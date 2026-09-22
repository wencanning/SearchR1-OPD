"""Frozen independent history probe; GPU4 yields immediately to other jobs."""
import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/independent_history_20260914'
MODEL='/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'
SEED='independent-history-20260914-v1'
GPU_UUID='GPU-209eafd7-38f5-a04a-12d5-8d316ea4f8a8'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_conflict(own_pid):
    try:
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                       '--format=csv,noheader,nounits'],text=True,timeout=3)
        own_memory=0; own_present=False
        for line in output.splitlines():
            uuid,pid,memory=[x.strip() for x in line.split(',')]
            if uuid!=GPU_UUID:continue
            if int(pid)!=own_pid:return f'GPU4 occupied by PID {pid}'
            own_present=True
            own_memory+=int(memory)
        used=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'],text=True,timeout=3).strip())
        # During our model transfer these two NVML snapshots are not atomic.
        # Never mistake our increasing allocation for an external job.
        if not own_present and used>256:return f'GPU4 unattributed memory {used} MiB'
    except Exception as exc:return f'GPU monitor failed: {type(exc).__name__}: {exc}'
    return None


def guard():
    reason=gpu_conflict(os.getpid())
    if reason:
        print('FALLBACK_CPU',reason,flush=True)
        # Only this diagnostic process exits; no signal is sent to any other PID.
        os._exit(75)


def watch(stop):
    while not stop.wait(2):guard()


def prepare(tok):
    import pandas as pd
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    labels={r['review_id']:r for r in json.loads((OUT/'annotations.json').read_text())}
    assert len(labels)==50
    frame=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    sources={(r['data_source'],int(r['extra_info']['index'])):r for r in frame.to_dict('records')}
    candidates=[]; excluded=[]
    for c in manifest['cases']:
        a=labels[c['review_id']]
        if not(c['score']==0 and a['evidence']=='sufficient' and a['answer_semantic']=='wrong' and a['history_error']=='yes'):continue
        reason=None
        if c['malformed_information']:reason='malformed information tags'
        if c['missing_answer']:reason='no complete answer'
        path=ROOT/c['source_file']
        assert sha(path)==manifest['source_hashes'][c['source_file']]
        with gzip.open(path,'rt') as f:
            next(f); r=next(r for line in f for r in [json.loads(line)] if int(r['index'])==c['index'])
        text=''.join(r['token_texts']); assert text==c['response']
        answer_matches=list(re.finditer(r'<answer>(.*?)</answer>',text,re.S))
        if not answer_matches:reason='no complete answer'
        # Balanced tags are required for the narrow mechanism probe, not for random-sample retention.
        for tag in ['think','information','search','answer']:
            if text.count('<'+tag+'>')!=text.count('</'+tag+'>'):reason=f'unbalanced {tag} tags'
        bounds=[]; pos=0
        for token in r['token_texts']:bounds.append((pos,pos+len(token)));pos+=len(token)
        first_info=text.find('<information>')
        post=[m for m in re.finditer(r'<think>(.*?)</think>',text,re.S) if m.start()>first_info>=0]
        claim_response_ids=set(); span_records=[]
        for snippet in a['spans']:
            assert text.count(snippet)==1
            start=text.index(snippet);end=start+len(snippet)
            thought=next((m for m in post if m.start(1)<=start and end<=m.end(1)),None)
            if thought is None:
                reason='annotated assertion outside post-search think body';continue
            ids=[i for i,(s,e) in enumerate(bounds) if s<end and e>start and s>=thought.start(1) and e<=thought.end(1)]
            assert ids and all(r['loss_mask'][i] for i in ids)
            claim_response_ids.update(ids);span_records.append(dict(text=snippet,indices=ids))
        if reason:
            excluded.append(dict(review_id=c['review_id'],key=c['key'],reason=reason));continue
        am=answer_matches[-1];start=am.start(1);end=am.end(1)
        while text[start].isspace():start+=1
        while text[end-1].isspace():end-=1
        cut=next(i for i,(s,e) in enumerate(bounds) if s<=start<e)
        prefix=''.join(r['token_texts'][:cut]);leading=text[bounds[cut][0]:start];closing=text[end:am.end()]
        source=sources[(r['data_source'],c['index'])]
        prompt_ids=tok.apply_chat_template(list(source['prompt']),add_generation_prompt=True,tokenize=True)
        ids=prompt_ids+r['token_ids'][:cut];offset=len(prompt_ids)
        if len(ids)>4096:
            excluded.append(dict(review_id=c['review_id'],key=c['key'],reason='prefix >4096 tokens'));continue
        broad={i for i,(s,e) in enumerate(bounds[:cut]) if r['loss_mask'][i] and any(m.start(1)<=s and e<=m.end(1) for m in post)}
        assert claim_response_ids and claim_response_ids<=broad
        original_labels=tok(leading+c['answer']+closing,add_special_tokens=False)['input_ids']
        assert original_labels==r['token_ids'][cut:cut+len(original_labels)]
        def mask(indices):return [0 if i-offset in indices else 1 for i in range(len(ids))]
        gold=c['gold'][0]
        candidate=dict(key=c['key'],review_id=c['review_id'],method=c['method'],question=c['question'],gold=gold,
            aliases=c['gold'],wrong=c['answer'],leading=leading,closing=closing,prefix_text=prefix,
            prefix_n=len(ids),claim_tokens=len(claim_response_ids),broad_tokens=len(broad),annotated_spans=span_records,
            original_saved_logps=r['teacher_log_prob'][cut:cut+len(original_labels)],
            original_answer_advantage_sum=sum(r['opd_advantage'][i] for i in range(cut,cut+len(original_labels))),
            conditions={name:dict(ids=ids,mask=mask(indices)) for name,indices in
                [('original',set()),('hide_claims',claim_response_ids),('hide_all_post',broad)]},
            generation_length_eligible=len(tok(leading+gold+closing,add_special_tokens=False)['input_ids'])<=16)
        candidates.append(candidate)
    chosen=[]
    for method in ['opd','sod']:
        subset=sorted([c for c in candidates if c['method']==method],key=lambda c:hashlib.sha256((SEED+'|probe|'+c['key']).encode()).hexdigest())
        chosen.extend(subset[:3]);selected_gen=False
        for c in subset[:3]:
            c['free_generate']=not selected_gen and c['generation_length_eligible'] and c['prefix_n']<=2048
            selected_gen |= c['free_generate']
        excluded.extend(dict(review_id=c['review_id'],key=c['key'],reason='predefined maximum 3 per method') for c in subset[3:])
    result=dict(sample_sha256=sha(OUT/'sample_manifest.json'),annotations_sha256=sha(OUT/'annotations.json'),
        cases=chosen,excluded=excluded)
    path=OUT/'probe_manifest.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('FROZEN',[(c['review_id'],c['key'],c['prefix_n'],c['claim_tokens'],c['free_generate']) for c in chosen],
          'EXCLUDED',excluded,flush=True)
    return chosen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','gpu4'],default='cpu');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='4' if args.device=='gpu4' else ''
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    cases=prepare(tok)
    if args.prepare_only or not cases:return
    stop=threading.Event()
    if args.device=='gpu4':
        guard();threading.Thread(target=watch,args=(stop,),daemon=True).start()
    device=torch.device('cuda:0' if args.device=='gpu4' else 'cpu')
    witness=torch.randn(8,8,device=device)
    assert torch.isfinite(witness@witness).all()
    print('KERNEL_WITNESS',str(device),torch.__version__,flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32,
        low_cpu_mem_usage=True,attn_implementation='eager').eval().to(device)
    print('LOADED',args.device,'PID',os.getpid(),'dtype',str(next(model.parameters()).dtype),flush=True)
    def forward(ids,mask,n):
        if args.device=='gpu4':guard()
        with torch.inference_mode():
            logits=model(input_ids=torch.tensor([ids],device=device),attention_mask=torch.tensor([mask],device=device),
                position_ids=torch.arange(len(ids),device=device)[None,:],use_cache=False,num_logits_to_keep=n).logits[0]
            assert torch.isfinite(logits).all()
            return logits.log_softmax(-1).cpu()
    def score(c,condition,answer):
        encoded=tok(c['leading']+answer+c['closing'],add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids'];end=len(c['leading']+answer)
        content=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>len(c['leading'])]
        lp=forward(condition['ids']+ids[:-1],condition['mask']+[1]*(len(ids)-1),len(ids))
        chosen=lp.gather(-1,torch.tensor(ids)[:,None]).squeeze(-1);body=chosen[content];top=int(lp[0].argmax())
        return dict(full_sum=chosen.sum().item(),full_mean=chosen.mean().item(),answer_sum=body.sum().item(),answer_mean=body.mean().item(),
            token_ids=ids,token_logps=chosen.tolist(),first_top_token=tok.decode([top]),first_top_id=top,first_top_logp=float(lp[0,top]))
    score_path=OUT/'probe_scores.jsonl';done={}
    if score_path.exists():done={r['key']:r for l in score_path.read_text().splitlines() for r in [json.loads(l)]}
    numerical_path=OUT/'numerical_exclusions.jsonl'
    numerical={json.loads(l)['key'] for l in numerical_path.read_text().splitlines()} if numerical_path.exists() else set()
    for c in cases:
        if c['key'] in done or c['key'] in numerical:continue
        started=time.time();print('START',c['review_id'],flush=True)
        base=score(c,c['conditions']['original'],c['wrong'])
        assert len(base['token_logps'])==len(c['original_saved_logps'])
        error=max(abs(a-b) for a,b in zip(base['token_logps'],c['original_saved_logps']))
        if error>=.5:
            with numerical_path.open('a') as f:f.write(json.dumps(dict(key=c['key'],max_error=error,device=args.device))+'\n')
            numerical.add(c['key']);print('EXCLUDED_NUMERIC',c['key'],error,flush=True);continue
        scores={}
        for name,condition in c['conditions'].items():
            scores[name]={candidate:base if name=='original' and candidate=='wrong' else score(c,condition,c[candidate]) for candidate in ['gold','wrong']}
            s=scores[name];assert s['gold']['first_top_id']==s['wrong']['first_top_id']
            assert abs(s['gold']['first_top_logp']-s['wrong']['first_top_logp'])<.001
            print('MARGIN',c['review_id'],name,s['gold']['full_sum']-s['wrong']['full_sum'],flush=True)
        row=dict(key=c['key'],review_id=c['review_id'],device=args.device,baseline_max_error=error,scores=scores,seconds=time.time()-started)
        with score_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        done[c['key']]=row
    gen_path=OUT/'free_generation.jsonl'
    generated={(r['key'],r['condition']) for l in gen_path.read_text().splitlines() for r in [json.loads(l)]} if gen_path.exists() else set()
    for c in cases:
        if not c['free_generate'] or c['key'] not in done:continue
        for name in ['original','hide_claims']:
            if (c['key'],name) in generated:continue
            condition=c['conditions'][name];new=[];reason='max_new_tokens';started=time.time()
            for t in range(16):
                lp=forward(condition['ids']+new,condition['mask']+[1]*len(new),1)[0]
                nxt=int(lp.argmax())
                if t==0:
                    assert nxt==done[c['key']]['scores'][name]['gold']['first_top_id']
                    assert abs(float(lp[nxt])-done[c['key']]['scores'][name]['gold']['first_top_logp'])<.001
                new.append(nxt);text=tok.decode(new,skip_special_tokens=False)
                if '</answer>' in text:reason='answer_closed';break
                if nxt==tok.eos_token_id:reason='eos';break
            text=tok.decode(new,skip_special_tokens=False)
            answer=text.split('</answer>')[0].strip() if reason=='answer_closed' else None
            from audit_pure_opd_mainline import norm
            exact=norm(answer) in [norm(g) for g in c['aliases']] if answer is not None else None
            row=dict(key=c['key'],review_id=c['review_id'],condition=name,device=args.device,token_ids=new,
                text=text,answer=answer,stop_reason=reason,exact_match=exact,seconds=time.time()-started)
            with gen_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print('GENERATED',json.dumps(row,ensure_ascii=False),flush=True)
    stop.set();print('COMPLETE',flush=True)


if __name__=='__main__':main()
