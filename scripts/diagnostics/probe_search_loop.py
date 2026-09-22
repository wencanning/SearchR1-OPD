"""Frozen common-prefix search-loop diagnosis. Never changes live training weights."""
import argparse
import bisect
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/search_loop_20260914'
TAG = '20260914-1220-coef1'
TEACHER = '/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'
SEED = 'search-loop-20260914-v1'
UUID = 'GPU-4491c0be-1ea7-eff5-5e5b-9e4fe635f053'


def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()


def append(name, row):
    with (OUT / name).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()


def gpu_check():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
    foreign = [int(p.strip()) for u, p in (l.split(',') for l in raw.splitlines())
               if u.strip() == UUID and int(p.strip()) not in [os.getpid(), 110019]]
    if foreign:
        raise RuntimeError(f'GPU3 unknown PIDs; do not displace them: {foreign}')


def rows(method, step):
    path = ROOT / f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
    with gzip.open(path, 'rt') as f:
        header = json.loads(next(f))
        result = [json.loads(l) for l in f]
    assert len(result) == 128 and header['metrics']['opd/grpo_advantage'] == 0
    return path, result


def norm(s):
    return ' '.join(s.lower().split())


def events(r):
    text = ''.join(r['token_texts'])
    ends = []; pos = 0
    for token in r['token_texts']:
        pos += len(token); ends.append(pos)
    def indices(a, b):
        return list(range(bisect.bisect_right(ends, a), bisect.bisect_left(ends, b) + 1))
    qs = [m for m in re.finditer(r'<search>(.*?)</search>', text, re.S)
          if all(r['loss_mask'][i] for i in indices(m.start(), m.end()))]
    result = []
    for k, q in enumerate(qs):
        stop = qs[k+1].start() if k+1 < len(qs) else len(text)
        for m in re.finditer(r'<information>(.*?)</information>', text[q.end():stop], re.S):
            a, b = q.end()+m.start(1), q.end()+m.end(1)
            ids = indices(a, b)
            if not ids or any(r['loss_mask'][i] for i in ids):
                continue
            end_idx = bisect.bisect_left(ends, q.end()+m.end()) + 1
            cut = next((i for i in range(end_idx, len(ends)) if r['loss_mask'][i]), None)
            if cut is not None:
                result.append(dict(query=q[1].strip(), observation=m[1], cut=cut, body_ids=ids, query_number=k))
            break
    return result


def prepare():
    import pandas as pd
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B', local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    assert tok.get_vocab() == teacher_tok.get_vocab(), 'Teacher/student vocab IDs differ'
    frame = pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    sources = {(r['data_source'], int(r['extra_info']['index'])): r for r in frame.to_dict('records')}
    files = []; cases = []; exclusions = []; replay = []
    def prompt(r):
        src = sources[(r['data_source'], int(r['index']))]
        return tok.apply_chat_template(list(src['prompt']), add_generation_prompt=True, tokenize=True), src['question']
    for method, step, kind, limit in [('opd',104,'repeat',32), ('sod',108,'repeat',32),
                                       ('opd',90,'control',16), ('sod',90,'control',16)]:
        path, rs = rows(method, step)
        files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        pool = []
        for r in rs:
            ev = events(r)
            pair = next(((a,b) for a,b in zip(ev,ev[1:]) if b['query_number']==a['query_number']+1
                         and norm(a['query'])==norm(b['query']) and a['observation']==b['observation']), None)
            if kind == 'repeat' and pair is None: continue
            if kind == 'control' and (not ev or len({norm(e['query']) for e in ev}) != len(ev)): continue
            selected = [('first_return', pair[0]), ('repeat_return', pair[1])] if pair else [('first_return', ev[0])]
            p, question = prompt(r)
            key = f'{method}:{step}:{r["index"]}'
            if max(len(p)+e['cut'] for _,e in selected)>3072:
                exclusions.append(dict(key=key, reason='prefix_over_3072')); continue
            states = []
            for name,e in selected:
                cut=e['cut']; ids=p+r['token_ids'][:cut]
                horizon=next((i for i in range(cut,len(r['loss_mask'])) if not r['loss_mask'][i]),len(r['loss_mask']))
                labels=r['token_ids'][cut:min(horizon,cut+32)]
                states.append(dict(name=name,ids=ids,mask=[1]*len(ids),query=e['query'],
                    seen_queries=[x['query'] for x in ev if x['cut']<=cut],
                    saved_next_ids=labels,saved_teacher_logps=r['teacher_log_prob'][cut:cut+len(labels)],
                    actual_next_text=''.join(r['token_texts'][cut:horizon]),
                    saved_student_logps=r['student_log_prob'][cut:cut+len(labels)]))
            if pair:
                last=dict(states[-1]); last['name']='hide_duplicate_observation'
                hidden={len(p)+i for i in pair[1]['body_ids']}
                last['mask']=[int(i not in hidden) for i in range(len(last['ids']))]
                states.append(last)
            pool.append(dict(key=key,method=method,step=step,kind=kind,index=r['index'],question=question,
                             score=r['sequence_score'],states=states))
        pool.sort(key=lambda c:digest(SEED+'|'+c['key']))
        cases.extend(pool[:limit])
        print('POOL',method,step,kind,len(pool),'selected',min(limit,len(pool)),flush=True)
    for method in ['opd','sod']:
        path,rs=rows(method,101)
        files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rs.sort(key=lambda r:digest(SEED+'|replay|'+method+str(r['index'])))
        selected=[]
        for r in rs:
            p,q=prompt(r)
            if len(p)+len(r['token_ids'])>3072:continue
            text=''.join(r['token_texts']); ends=[];pos=0
            for token in r['token_texts']:pos+=len(token);ends.append(pos)
            tags=set()
            for m in re.finditer(r'<(think|search|answer)>',text):
                tags.update(range(bisect.bisect_right(ends,m.start()),bisect.bisect_left(ends,m.end())+1))
            selected.append(dict(key=f'{method}:101:{r["index"]}',method=method,prompt_ids=p,
                token_ids=r['token_ids'],loss_mask=r['loss_mask'],old_logps=r['student_log_prob'],
                advantages=r['weighted_opd_advantage'],tag_mask=[int(i in tags and r['loss_mask'][i]) for i in range(len(r['token_ids']))]))
            if len(selected)==8:break
        replay.extend(selected)
    result=dict(seed=SEED,cases=cases,files=files,exclusions=exclusions,replay=replay)
    target=OUT/'manifest.json'
    if target.exists():assert json.loads(target.read_text())==result, 'Frozen manifest changed'
    else:target.write_text(json.dumps(result,ensure_ascii=False))
    print('FROZEN',len(cases),'cases',sum(len(c['states']) for c in cases),'states',flush=True)


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu_check()
    witness=torch.randn(8,8,device='cuda');assert torch.isfinite(witness@witness).all()
    print('KERNEL_WITNESS',torch.__version__,torch.cuda.get_device_name(),flush=True)
    manifest=json.loads((OUT/'manifest.json').read_text())
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    paths={'teacher_fp16':TEACHER,'teacher':TEACHER,'initial':str(ROOT/'data/student/0.5B')}
    for method in ['opd','sod']:
        for step in [75,100]:paths[f'{method}{step}']=str(ROOT/f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/actor/global_step_{step}')
    metadata=dict(pid=os.getpid(),started=time.time(),models=paths,manifest_sha256=digest((OUT/'manifest.json').read_text()),
                  torch=torch.__version__,dtype='FP32 plus training-matched FP16 teacher',attention='sdpa',max_new_tokens=256,
                  output_version=2,logit_vocab_size=len(tok))
    if (OUT/'run_metadata.json').exists():append('run_metadata_history.jsonl',json.loads((OUT/'run_metadata.json').read_text()))
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    models={}
    for name,path in paths.items():
        gpu_check()
        models[name]=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16 if name=='teacher_fp16' else torch.float32,
            low_cpu_mem_usage=True,attn_implementation='sdpa').eval().to('cuda')
        print('LOADED',name,'allocated_GiB',round(torch.cuda.memory_allocated()/2**30,2),flush=True)
    def check():
        if (OUT/'STOP').exists():raise RuntimeError('STOP requested')
        gpu_check()
    def logps(model,state,labels):
        ids=state['ids']+labels[:-1]; mask=state['mask']+[1]*(len(labels)-1)
        with torch.inference_mode():
            logits=model(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.tensor([mask],device='cuda'),
                position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=False,num_logits_to_keep=len(labels)).logits[0]
            assert torch.isfinite(logits).all()
            lp=logits[:,:len(tok)].float().log_softmax(-1).gather(-1,torch.tensor(labels,device='cuda')[:,None]).squeeze(-1)
            return lp.cpu().tolist()
    def action_scores(model,state):
        scores={};length=len(state['ids'])
        with torch.inference_mode():
            output=model(input_ids=torch.tensor([state['ids']],device='cuda'),attention_mask=torch.tensor([state['mask']],device='cuda'),
                position_ids=torch.arange(length,device='cuda')[None],use_cache=True,num_logits_to_keep=1)
            cache=output.past_key_values
            if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
            first=output.logits[0,-1,:len(tok)].float().log_softmax(-1)
            for prefix in ['', ' ', '\n', '\n\n']:
                for tag in ['think','search','answer']:
                    word=prefix+'<'+tag+'>';labels=tok(word,add_special_tokens=False)['input_ids']
                    value=float(first[labels[0]])
                    if len(labels)>1:
                        tail=model(input_ids=torch.tensor([labels[:-1]],device='cuda'),
                            attention_mask=torch.tensor([state['mask']+[1]*(len(labels)-1)],device='cuda'),
                            position_ids=torch.arange(length,length+len(labels)-1,device='cuda')[None],
                            past_key_values=cache,use_cache=True,num_logits_to_keep=len(labels)-1).logits[0,:,:len(tok)].float()
                        value+=float(tail.log_softmax(-1).gather(-1,torch.tensor(labels[1:],device='cuda')[:,None]).sum())
                    assert cache[0][0].shape[-2]==length, 'Candidate mutated shared prefix KV'
                    scores[word]=value
            # Cache branch is only a speed optimization; validate against an uncached call per state.
            labels=tok('<search>',add_special_tokens=False)['input_ids']
            uncached=sum(logps(model,state,labels))
            tolerance=.15 if next(model.parameters()).dtype==torch.float16 else .002
            cache_error=abs(uncached-scores['<search>'])
            if cache_error>=tolerance:
                append('cache_fallbacks.jsonl',dict(state=state['name'],prefix_sha256=digest(str(state['ids'])),
                    dtype=str(next(model.parameters()).dtype),error=cache_error,tolerance=tolerance,
                    uncached_search=uncached,cached_search=scores['<search>'],action='recompute_all_candidates_uncached'))
                scores={word:sum(logps(model,state,tok(word,add_special_tokens=False)['input_ids'])) for word in scores}
                print('UNCACHED_FALLBACK',state['name'],cache_error,flush=True)
        return scores
    def classify(text,state):
        a=re.search(r'<(search|answer)>(.*?)</\1>',text,re.S)
        if a is None:return dict(action='incomplete',repeat=None)
        query=a[2].strip()
        return dict(action=a[1],value=query,repeat=(norm(query) in {norm(q) for q in state['seen_queries']}) if a[1]=='search' else False)
    def generate_batch(model,items,seed):
        # Custom KV-cached decoding stops at the first complete action, not model EOS alone.
        width=max(len(s['ids']) for _,s in items)
        ids=torch.tensor([[tok.pad_token_id]*(width-len(s['ids']))+s['ids'] for _,s in items],device='cuda')
        masks=torch.tensor([[0]*(width-len(s['mask']))+s['mask'] for _,s in items],device='cuda')
        positions=torch.tensor([[0]*(width-len(s['ids']))+list(range(len(s['ids']))) for _,s in items],device='cuda')
        rng=[]
        for c,s in items:
            g=torch.Generator(device='cuda');g.manual_seed(int(digest(f'{seed}|{c["key"]}|{s["name"]}')[:12],16));rng.append(g)
        new=[[] for _ in items];done=[False]*len(items);cache=None
        with torch.inference_mode():
            for t in range(256):
                if t%32==0:check()
                output=model(input_ids=ids,attention_mask=masks,position_ids=positions,past_key_values=cache,
                             use_cache=True,num_logits_to_keep=1)
                logits=output.logits[:,-1,:len(tok)].float();assert torch.isfinite(logits).all();cache=output.past_key_values
                if seed is None:nxt=logits.argmax(-1)
                else:nxt=torch.stack([torch.multinomial(logits[j].softmax(-1),1,generator=rng[j])[0] for j in range(len(items))])
                for j,(_,state) in enumerate(items):
                    if done[j]:continue
                    token=int(nxt[j]);new[j].append(token)
                    text=tok.decode(new[j],skip_special_tokens=False)
                    done[j]=token==tok.eos_token_id or classify(text,state)['action']!='incomplete'
                if all(done):break
                ids=nxt[:,None];masks=torch.cat([masks,torch.ones((len(items),1),device='cuda',dtype=masks.dtype)],1)
                positions=torch.tensor([[len(s['ids'])+t] for _,s in items],device='cuda')
        return [dict(text=tok.decode(tokens,skip_special_tokens=False),new_tokens=len(tokens),
                     **classify(tok.decode(tokens,skip_special_tokens=False),s)) for tokens,(_,s) in zip(new,items)]
    cases=manifest['cases']; scored=set();generated=set()
    if (OUT/'scores_v2.jsonl').exists():scored={(r['model'],r['key'],r['state']) for l in (OUT/'scores_v2.jsonl').read_text().splitlines() for r in [json.loads(l)]}
    if (OUT/'generations_v2.jsonl').exists():generated={(r['model'],r['key'],r['state'],r['seed']) for l in (OUT/'generations_v2.jsonl').read_text().splitlines() for r in [json.loads(l)]}
    # A small, source-balanced pilot finishes before the expanded run.
    pilot=[]
    for method in ['opd','sod']:
        for kind in ['repeat','control']:pilot.extend([c for c in cases if c['method']==method and c['kind']==kind][:8])
    stages=[('pilot',pilot,None),('full',cases,None)]
    def evaluate(stage,subset,seed):
        for name,model in models.items():
            check();print('STAGE',stage,name,'seed',seed,flush=True)
            for c in subset:
                for state in c['states']:
                    key=(name,c['key'],state['name'])
                    if key in scored or seed is not None:continue
                    score=action_scores(model,state)
                    row=dict(model=name,key=c['key'],kind=c['kind'],source_method=c['method'],state=state['name'],scores=score)
                    if name in ['teacher','teacher_fp16'] and state['name']!='hide_duplicate_observation':
                        fresh=logps(model,state,state['saved_next_ids'])
                        error=max(abs(a-b) for a,b in zip(fresh,state['saved_teacher_logps']))
                        row.update(saved_teacher_max_error=error,numerical_pass=error<=.5,fresh_logps=fresh)
                    append('scores_v2.jsonl',row);scored.add(key)
            pending=[(c,s) for c in subset for s in c['states'] if (name,c['key'],s['name'],seed) not in generated]
            pending.sort(key=lambda x:len(x[1]['ids']))
            batch_size=8
            while pending:
                check();batch=pending[:batch_size];started=time.time()
                try:outputs=generate_batch(model,batch,seed)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size==1:raise
                    batch_size=max(1,batch_size//2);print('OOM_RETRY_BATCH',batch_size,flush=True);continue
                for (c,s),output in zip(batch,outputs):
                    append('generations_v2.jsonl',dict(model=name,key=c['key'],kind=c['kind'],source_method=c['method'],
                        state=s['name'],seed=seed,**output));generated.add((name,c['key'],s['name'],seed))
                del pending[:len(batch)]
                print('GENERATED',stage,name,seed,len(batch),'remaining',len(pending),'seconds',round(time.time()-started,2),
                      'reserved_GiB',round(torch.cuda.memory_reserved()/2**30,2),flush=True)
        append('milestones.jsonl',dict(stage=stage,seed=seed,completed=time.time()))
    for stage,subset,seed in stages:evaluate(stage,subset,seed)
    # Controlled one-step diagnostic: fresh optimizer, NOT exact historical training replay.
    from verl.trainer.ppo.core_algos import compute_policy_loss
    for method in ['opd','sod']:
        check();base=models[method+'100'];saved={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
        examples=[r for r in manifest['replay'] if r['method']==method]
        probe_states=[(c,c['states'][1]) for c in cases if c['method']==method and c['kind']=='repeat'][:8]
        for variant in ['full_policy','opening_tags']:
            base.load_state_dict(saved);base.eval()
            before={c['key']:action_scores(base,s) for c,s in probe_states}
            fresh_old=[]
            # Only policy positions are projected into the full vocabulary, keeping memory bounded.
            def selected_logps(r,grad):
                ids=r['prompt_ids']+r['token_ids'];p=len(r['prompt_ids'])
                selected=[i for i,m in enumerate(r['loss_mask']) if m]
                hidden=base.model(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long),
                    position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=False).last_hidden_state[0]
                values=[]
                for start in range(0,len(selected),128):
                    inds=selected[start:start+128]
                    logits=base.lm_head(hidden[torch.tensor([p+i-1 for i in inds],device='cuda')])[:,:len(tok)].float()
                    values.append(logits.log_softmax(-1).gather(-1,torch.tensor([r['token_ids'][i] for i in inds],device='cuda')[:,None]).squeeze(-1))
                return torch.cat(values),selected
            with torch.no_grad():
                for r in examples:
                    values,inds=selected_logps(r,False)
                    fresh_old.extend(abs(float(v)-r['old_logps'][i]) for v,i in zip(values,inds))
            if max(fresh_old)>.5:
                append('updates.jsonl',dict(method=method,variant=variant,status='skipped_parity',max_old_logp_error=max(fresh_old)));continue
            base.train();base.gradient_checkpointing_enable()
            optimizer=torch.optim.AdamW(base.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01)
            optimizer.zero_grad();loss_sum=0
            for r in examples:
                values,inds=selected_logps(r,True)
                old=torch.tensor([r['old_logps'][i] for i in inds],device='cuda')[None]
                adv=torch.tensor([r['advantages'][i]*(1 if variant=='full_policy' else r['tag_mask'][i]) for i in inds],device='cuda')[None]
                loss,*_=compute_policy_loss(old,values[None],adv,torch.ones_like(old),.2,.2,.28,3.)
                (loss/len(examples)).backward();loss_sum+=float(loss)/len(examples)
            grad=float(torch.nn.utils.clip_grad_norm_(base.parameters(),1.));optimizer.step();optimizer.zero_grad(set_to_none=True)
            base.eval();base.gradient_checkpointing_disable()
            after={c['key']:action_scores(base,s) for c,s in probe_states}
            append('updates.jsonl',dict(method=method,variant=variant,status='complete',fresh_optimizer=True,
                historical_exact_replay=False,trajectories=len(examples),max_old_logp_error=max(fresh_old),loss=loss_sum,
                grad_norm=grad,before=before,after=after))
            del optimizer
            print('UPDATE_COMPLETE',method,variant,'grad',grad,flush=True)
        base.load_state_dict(saved);base.eval();base.gradient_checkpointing_disable();del saved
    append('milestones.jsonl',dict(stage='updates',completed=time.time()))
    for seed in [42,43,44]:evaluate('sampled',cases,seed)
    append('milestones.jsonl',dict(stage='ALL_COMPLETE',completed=time.time()))
    print('ALL_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='' if args.prepare else '3'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['OMP_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    if args.prepare:prepare()
    else:run()
