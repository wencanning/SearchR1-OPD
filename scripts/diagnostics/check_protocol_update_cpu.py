"""Small full-network one-step causal screen, CPU, fresh optimizer, no live edits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from probe_search_loop import ROOT,rows,events
from verl.trainer.ppo.core_algos import compute_policy_loss

OUT=ROOT/'reports/action-credit-recovery-20260914'
MODEL_PATH=ROOT/'verl_checkpoints/pure-opd-05B-n1-20260914-1220-coef1/actor/global_step_75'
torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
frame=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
lookup={(r['data_source'],int(r['extra_info']['index'])):r for r in frame.to_dict('records')}
source,rs=rows('opd',76)
rs.sort(key=lambda r:hashlib.sha256(f'protocol-network-update-v1|{r["index"]}'.encode()).hexdigest())
cases=[]
for r in rs:
    if r['token_ids'][:2]!=[13708,766]:continue
    data=lookup[(r['data_source'],int(r['index']))]
    prompt=tok.apply_chat_template(list(data['prompt']),add_generation_prompt=True,tokenize=True)
    if len(prompt)+len(r['token_ids'])>3072:continue
    ev=events(r)
    if not ev:continue
    cases.append(dict(key=f'opd:76:{r["index"]}',prompt=prompt,ids=r['token_ids'],mask=r['loss_mask'],
        tags=[s=='tag' for s in r['segments']],old=r['student_log_prob'],adv=r['opd_advantage'],post_cut=ev[0]['cut']))
    if len(cases)==8:break
assert len(cases)==8
(OUT/'cpu_update_manifest.json').write_text(json.dumps(dict(cases=cases,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    initial_checkpoint=str(MODEL_PATH),fresh_optimizer=True,exact_historical_replay=False),ensure_ascii=False))
model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,torch_dtype=torch.float32,
    low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
initial={k:v.detach().clone() for k,v in model.state_dict().items()}
@torch.autocast(device_type='cpu',dtype=torch.bfloat16)
def policy_logps(c):
    ids=c['prompt']+c['ids'];n=len(c['prompt']);selected=[i for i,m in enumerate(c['mask']) if m]
    hidden=model.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
        position_ids=torch.arange(len(ids))[None],use_cache=False).last_hidden_state[0]
    result=[]
    for start in range(0,len(selected),64):
        subset=selected[start:start+64]
        logits=model.lm_head(hidden[torch.tensor([n+i-1 for i in subset])])[:,:len(tok)].float()
        result.append(logits.log_softmax(-1).gather(-1,torch.tensor([c['ids'][i] for i in subset])[:,None]).squeeze(-1))
    return torch.cat(result),selected
@torch.autocast(device_type='cpu',dtype=torch.bfloat16)
def probe():
    result={}
    with torch.no_grad():
        for c in cases:
            result[c['key']]={}
            for name,ids in [('initial',c['prompt']),('post',c['prompt']+c['ids'][:c['post_cut']])]:
                logits=model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
                    position_ids=torch.arange(len(ids))[None],use_cache=False,num_logits_to_keep=1).logits[0,-1,:len(tok)]
                lp=logits.float().log_softmax(-1)
                result[c['key']][name]=dict(canonical=float(lp[13708]),split=float(lp[27]))
    return result
errors=[]
with torch.no_grad():
    for c in cases:
        lp,selected=policy_logps(c);errors.extend(abs(float(a)-c['old'][i]) for a,i in zip(lp,selected))
        c['replay_old']={i:float(v) for i,v in zip(selected,lp)}
        c['replay_adv']={i:c['old'][i]+c['adv'][i]-float(v) for i,v in zip(selected,lp)}
print('PARITY',max(errors),flush=True)
assert max(errors)<.5, 'Checkpoint75 vs step76 old probabilities mismatch; no update'
for variant in ['full_policy','mask_protocol_tags','mask_initial_think_only']:
    start=time.time();model.load_state_dict(initial);model.eval();before=probe()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01)
    optimizer.zero_grad();model.train();loss_value=0
    for c in cases:
        lp,selected=policy_logps(c)
        old=torch.tensor([c['replay_old'][i] for i in selected])[None]
        def keep(i):
            if variant=='mask_protocol_tags':return not c['tags'][i]
            if variant=='mask_initial_think_only':return i not in [0,1,2]
            return True
        adv=torch.tensor([c['replay_adv'][i]*keep(i) for i in selected])[None]
        loss,*_=compute_policy_loss(old,lp[None],adv,torch.ones_like(old),.2,.2,.28,3.)
        (loss/len(cases)).backward();loss_value+=float(loss)/len(cases)
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));optimizer.step();optimizer.zero_grad(set_to_none=True)
    model.eval();after=probe()
    row=dict(variant=variant,grad_norm=grad,loss=loss_value,before=before,after=after,seconds=time.time()-start,
             max_old_logp_error=max(errors),fresh_optimizer=True,exact_historical_replay=False,
             forward_dtype='autocast_bfloat16',ratio_old_recomputed_for_exact_unit_initial_ratio=True,
             normalization='mean over original policy tokens within trajectory, then mean over 8 trajectories')
    with (OUT/'cpu_updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print('UPDATE',variant,'grad',grad,'seconds',row['seconds'],'mean_initial_logp_delta',
        sum(after[k]['initial']['canonical']-before[k]['initial']['canonical'] for k in before)/len(before),flush=True)
    del optimizer
print('CPU_UPDATES_COMPLETE',flush=True)
