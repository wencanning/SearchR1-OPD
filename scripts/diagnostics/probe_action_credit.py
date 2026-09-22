"""Research follow-up: raw protocol token paths and local KL gradient diagnostics.

This measures a single-state logit-space derivative, NOT a parameter update or
proof of the historical training cause. All training checkpoints remain frozen.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TEACHER='/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'
UUIDS={2:'GPU-79f08c8c-6fe7-4ab0-d16f-00ab0dba95df',3:'GPU-4491c0be-1ea7-eff5-5e5b-9e4fe635f053'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[2,3],required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--student-dtype',choices=['bfloat16','float32'],default='bfloat16');args=parser.parse_args()
    assert all(c.isalnum() or c in '-_' for c in args.run_id)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='8'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'reports'/args.run_id;out.mkdir(exist_ok=True,parents=True)
    assert not (out/'metadata.json').exists(), 'Use a fresh run id; never mix precisions or overwrite data'
    def append(name,row):
        with (out/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def check():
        if (ROOT/'experiment_logs/gpu23-autoloop-20260914/STOP').exists():raise RuntimeError('STOP requested')
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
        foreign=[int(p.strip()) for u,p in (line.split(',') for line in raw.splitlines())
                 if u.strip()==UUIDS[args.gpu] and int(p.strip()) not in [110019,os.getpid()]]
        if foreign:raise RuntimeError(f'Foreign GPU process, no displacement: {foreign}')
    check()
    manifest_path=ROOT/'reports/search_loop_20260914/manifest.json'
    manifest=json.loads(manifest_path.read_text());tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    paths={'teacher_fp16':TEACHER,'teacher_fp32':TEACHER,'initial':str(ROOT/'data/student/0.5B')}
    for method in ['opd','sod']:
        for step in [75,100]:paths[f'{method}{step}']=str(ROOT/f'verl_checkpoints/pure-{method}-05B-n1-20260914-1220-coef1/actor/global_step_{step}')
    metadata=dict(gpu=args.gpu,pid=os.getpid(),started=time.time(),models=paths,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        interpretation='local logit gradients, not historical parameter updates',vocab_size=len(tok),
        student_dtype=args.student_dtype,teacher_dtypes=['float16','float32'])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    models={}
    for name,path in paths.items():
        check();models[name]=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,low_cpu_mem_usage=True,
            torch_dtype=(torch.float16 if name=='teacher_fp16' else torch.float32 if name=='teacher_fp32'
                         else getattr(torch,args.student_dtype)),attn_implementation='sdpa').eval().to('cuda')
        print('LOADED',name,'GiB',round(torch.cuda.memory_allocated()/2**30,2),flush=True)
    def first(model,ids,cache=False,past=None):
        total=len(ids)+(past[0][0].shape[-2] if past is not None else 0)
        with torch.inference_mode():
            output=model(input_ids=torch.tensor([ids],device='cuda'),
                attention_mask=torch.ones((1,total),device='cuda',dtype=torch.long),
                position_ids=torch.arange(total-len(ids),total,device='cuda')[None],
                use_cache=cache,past_key_values=past,num_logits_to_keep=1)
        return output.logits[0,-1,:len(tok)].float().log_softmax(-1),output.past_key_values
    def branch_score(model,ids,labels):
        with torch.inference_mode():
            x=ids+labels[:-1]
            logits=model(input_ids=torch.tensor([x],device='cuda'),attention_mask=torch.ones((1,len(x)),device='cuda',dtype=torch.long),
                position_ids=torch.arange(len(x),device='cuda')[None],use_cache=False,num_logits_to_keep=len(labels)).logits[0,:,:len(tok)].float()
            return float(logits.log_softmax(-1).gather(-1,torch.tensor(labels,device='cuda')[:,None]).sum())
    parity_path=ROOT/'reports/action-credit-recovery-20260914/cpu_update_manifest.json'
    if parity_path.exists():
        for c in json.loads(parity_path.read_text())['cases']:
            ids=c['prompt']+c['ids'];n=len(c['prompt']);selected=[i for i,m in enumerate(c['mask']) if m]
            with torch.inference_mode():
                model=models['opd75'];hidden=model.model(input_ids=torch.tensor([ids],device='cuda'),
                    attention_mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long),
                    position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=False).last_hidden_state[0]
                errors=[]
                for start in range(0,len(selected),64):
                    ix=selected[start:start+64]
                    lp=model.lm_head(hidden[torch.tensor([n+i-1 for i in ix],device='cuda')])[:,:len(tok)].float().log_softmax(-1)
                    got=lp.gather(-1,torch.tensor([c['ids'][i] for i in ix],device='cuda')[:,None]).flatten().tolist()
                    errors.extend(abs(v-c['old'][i]) for v,i in zip(got,ix))
                append('student_parity.jsonl',dict(key=c['key'],max_error=max(errors),mean_error=sum(errors)/len(errors),
                    numerical_pass=max(errors)<.5,dtype=args.student_dtype,reference='actual step76 old log probabilities'))
                print('STUDENT_PARITY',c['key'],max(errors),flush=True)
                del hidden,lp
    focus_words=['search','think','answer']
    focus_ids=[tok.encode(w,add_special_tokens=False) for w in focus_words]
    assert all(len(x)==1 for x in focus_ids)
    focus=torch.tensor([x[0] for x in focus_ids],device='cuda')
    states=[(c,next(s for s in c['states'] if s['name']==('repeat_return' if c['kind']=='repeat' else 'first_return')))
            for c in manifest['cases']]
    for number,(case,state) in enumerate(states):
        check();print('CASE',number+1,len(states),case['key'],flush=True)
        ids=state['ids'];distributions={}
        for name,model in models.items():
            lp,cache=first(model,ids,cache=True)
            if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
            top=lp.topk(12)
            # Raw IDs expose alternate tokenizations such as '<'+'think'+'>' vs '<th'+'ink'+'>'.
            generated=[];running_lp=lp
            for _ in range(8):
                token=int(running_lp.argmax());generated.append(token)
                if token==tok.eos_token_id or '>' in tok.decode(generated):break
                running_lp,cache=first(model,[token],cache=True,past=cache)
            alternatives={}
            for word in focus_words:
                paths_ids={tuple(tok.encode('<'+word+'>',add_special_tokens=False)),
                           tuple(tok.encode('<',add_special_tokens=False)+tok.encode(word,add_special_tokens=False)+tok.encode('>',add_special_tokens=False))}
                variants=[dict(ids=list(labels),logp=branch_score(model,ids,list(labels))) for labels in sorted(paths_ids)]
                alternatives[word]=dict(paths=variants,known_paths_log_mass=float(torch.logsumexp(torch.tensor([v['logp'] for v in variants]),0)))
            after_lt,_=first(model,ids+tok.encode('<',add_special_tokens=False))
            distributions[name]=after_lt
            append('protocol_paths.jsonl',dict(key=case['key'],kind=case['kind'],model=name,
                top_ids=top.indices.tolist(),top_tokens=[tok.decode([i]) for i in top.indices.tolist()],top_logps=top.values.tolist(),
                greedy_opening_ids=generated,greedy_opening_text=tok.decode(generated),candidate_paths=alternatives,
                after_lt_focus_logps=dict(zip(focus_words,after_lt[focus].tolist()))))
        # Exact formulas include the normalization term. Never equate one sampled sign with net parameter motion.
        for teacher in ['teacher_fp16','teacher_fp32']:
            lq=distributions[teacher];q=lq.exp()
            for student in ['initial','opd75','opd100','sod75','sod100']:
                lp=distributions[student];p=lp.exp();kl=(p*(lp-lq)).sum()
                reverse=p*(lp-lq-kl);forward=p-q
                seed=int(hashlib.sha256((case['key']+'|'+student+'|'+teacher).encode()).hexdigest()[:12],16)
                rng=torch.Generator(device='cuda');rng.manual_seed(seed)
                samples=torch.multinomial(p,32768,replacement=True,generator=rng)
                adv=lq[samples]-lp[samples]
                component=adv[:,None]*(p[focus][None,:]-(samples[:,None]==focus[None,:]).float())
                monte={}
                for n in [1,8,32]:
                    means=component.reshape(-1,n,3).mean(1)
                    monte[str(n)]=dict(mean=means.mean(0).tolist(),std=means.std(0).tolist(),
                        positive_fraction=(means>0).float().mean(0).tolist(),replicates=len(means))
                y=focus[0];sampled_search=(lq[y]-lp[y])*(p[focus]-(focus==y).float())
                append('local_gradients.jsonl',dict(key=case['key'],kind=case['kind'],teacher=teacher,student=student,
                    focus=focus_words,teacher_prob=q[focus].tolist(),student_prob=p[focus].tolist(),reverse_kl=float(kl),
                    reverse_exact=reverse[focus].tolist(),forward_exact=forward[focus].tolist(),
                    sampled_search_gradient=sampled_search.tolist(),sampled_search_advantage=float(lq[y]-lp[y]),
                    monte_carlo=monte,note='positive gradient lowers that logit under direct gradient descent; not a full-network update'))
        del distributions
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=time.time(),states=len(states))))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
