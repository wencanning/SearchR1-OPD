"""Bounded frozen-rollout update, NOT a full online OPD trainer.

See reports/recovery_20260917_native/LOCAL_UPDATE_PROTOCOL.md before interpreting results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from probe import ROOT, GPU, STUDENT, save


def flatten(row):
    ids=list(row['initial_ids']);positions=[];targets=[]
    for seg in row['segments']:
        assert len(seg['ids'])==len(seg['loss_mask'])
        for token,mask in zip(seg['ids'],seg['loss_mask']):
            if mask:positions.append(len(ids)-1);targets.append(token)
            ids.append(token)
    assert positions and min(positions)>=0
    return ids[:positions[-1]+1],positions,targets


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=['base','student_replay','teacher_repair'],default='base')
    p.add_argument('--witness',action='store_true');args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and os.environ.get('RECOVERY_GPU_GUARDED')=='1'
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(2026091711)
    torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
    args.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (args.out/'status.json').exists(),'Use a new directory; optimizer restart is not supported'
    tok=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(STUDENT,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda')
    initial_bf16={k:v.detach().to(dtype=torch.bfloat16).cpu().clone() for k,v in model.named_parameters()}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.0,foreach=False)

    def saved_parameter_change():
        changed=0;total=0
        for name,value in model.named_parameters():
            after=value.detach().to(dtype=torch.bfloat16).cpu()
            changed+=int((after!=initial_bf16[name]).sum());total+=after.numel()
        return dict(bf16_changed_parameters=changed,total_parameters=total,bf16_changed_fraction=changed/total)

    def log_probs(row,grad=True):
        ids,pos,targets=flatten(row);x=torch.tensor([ids],device='cuda');ys=torch.tensor(targets,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=model.model(x,attention_mask=torch.ones_like(x),use_cache=False).last_hidden_state[0]
            selected=hidden[torch.tensor(pos,device='cuda')];parts=[]
            def head(h,y):
                z=model.lm_head(h)[:,:len(tok)].float()
                return z.log_softmax(-1).gather(1,y[:,None])[:,0]
            for start in range(0,len(pos),32):
                h=selected[start:start+32];y=ys[start:start+32]
                parts.append(checkpoint(head,h,y,use_reentrant=False) if grad else head(h,y))
        return torch.cat(parts)

    if args.witness:
        prefix=tok.encode('Question: What is the capital of France?\n',add_special_tokens=False)
        ids=tok.encode('<think>The capital is Paris.</think><answer>Paris</answer>',add_special_tokens=False)
        row=dict(initial_ids=prefix,segments=[dict(ids=ids,loss_mask=[1]*len(ids))])
        model.train();before=model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        loss=-log_probs(row).mean();assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
        delta=(model.model.layers[0].self_attn.q_proj.weight.detach()-before).abs().max().item();assert delta>0
        result=dict(state='complete',witness=True,loss=loss.item(),grad_norm=norm.item(),max_parameter_change=delta,peak_cuda_allocated=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        result.update(saved_parameter_change())
        save(args.out/'status.json',result);print('TRAIN_WITNESS',json.dumps(result),flush=True);return

    root=ROOT/'reports/recovery_20260917_native';summary=json.loads((root/'recovery_summary.json').read_text())
    assert summary['teacher_rescued_questions']>=4 and summary['verified_student_suffixes']>=4,'Qualification diversity gate failed'
    assert json.loads((root/'teacher_scores/status.json').read_text())['state']=='complete'
    sources=[root/'continue_student/continuations.jsonl',root/'teacher_scores/scores.jsonl',root/'verified_student_train.jsonl',root/'verified_teacher_train.jsonl',root/'LOCAL_UPDATE_PROTOCOL.md']
    all_student=[json.loads(x) for x in sources[0].read_text().splitlines()]
    base=[r for r in all_student if r['split']=='train']
    scores={(r['qid'],r['replicate']):r for r in map(json.loads,sources[1].read_text().splitlines())}
    student=[json.loads(x) for x in sources[2].read_text().splitlines()]
    teacher=[json.loads(x) for x in sources[3].read_text().splitlines()]
    score_manifest=json.loads((root/'teacher_scores/manifest.frozen.json').read_text())
    assert score_manifest['source_sha256']==hashlib.sha256(sources[0].read_bytes()).hexdigest()
    inputs=json.loads((root/'train_inputs.frozen.json').read_text())
    train_qs={r['qid'] for r in inputs['cases'] if r['split']=='train'}
    states={r['qid']:r for r in map(json.loads,(root/'collected/states.jsonl').read_text().splitlines())}
    for r in base+student+teacher:
        assert r['split']=='train' and r['qid'] in train_qs
        assert r['initial_ids']==states[r['qid']]['prefix_ids']
    assert all(r['em'] for r in student+teacher)
    eval_qs={r['qid'] for r in all_student if r['split']=='eval'}
    assert not eval_qs & {r['qid'] for r in base+student+teacher}
    aux={'base':base,'student_replay':student,'teacher_repair':teacher}[args.arm]
    manifest=dict(arm=args.arm,checkpoint=str(STUDENT),steps=8,lr=1e-6,seed=2026091711,base_per_step=4,aux_per_step=0 if args.arm=='base' else 2,loss_divisor=6,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(args.out/'manifest.frozen.json',manifest)
    if args.arm=='base':
        zero=args.out/'zero_step_checkpoint'
        state={k:v.detach().to(dtype=torch.bfloat16).cpu() for k,v in model.state_dict().items()}
        model.save_pretrained(zero,state_dict=state,safe_serialization=True);tok.save_pretrained(zero);del state
    # Compute initial probabilities using exactly the numerical implementation used for updating.
    old={};model.eval()
    with torch.no_grad():
        for r in base:
            key=(r['qid'],r['replicate']);_,pos,ys=flatten(r)
            assert scores[key]['positions']==pos and scores[key]['targets']==ys
            old[key]=log_probs(r,False).detach().cpu()
    rng=random.Random(2026091711);base_order=list(range(len(base)));rng.shuffle(base_order)
    aux_order=list(range(len(aux)));rng.shuffle(aux_order)
    model.train();total_policy=0;total_context=0;start_time=time.monotonic()
    for step in range(8):
        opt.zero_grad(set_to_none=True);losses=[];entries=[]
        selected=[(base[base_order[(step*4+j)%len(base)]],'opd') for j in range(4)]
        if args.arm!='base':selected +=[(aux[aux_order[(step*2+j)%len(aux)]],'ce') for j in range(2)]
        for row,kind in selected:
            lp=log_probs(row);key=(row['qid'],row['replicate'])
            if kind=='opd':
                oldlp=old[key].to('cuda');tlp=torch.tensor(scores[key]['teacher_log_probs'],device='cuda')
                advantage=tlp-oldlp;ratio=(lp-oldlp).exp()
                if step==0:
                    error=(ratio.detach()-1).abs().max().item()
                    print('INITIAL_RATIO_MAX_ERROR',error,flush=True)
                    assert error<.02,'Initial ratio is not numerically near1'
                loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
            else:loss=-lp.mean()
            assert torch.isfinite(loss);(loss/6).backward();losses.append(loss.detach().item())
            ids,pos,_=flatten(row);total_policy+=len(pos);total_context+=len(ids)
            entries.append(dict(qid=row['qid'],replicate=row['replicate'],kind=kind,policy_tokens=len(pos),context_tokens=len(ids)))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
        record=dict(step=step+1,loss=sum(losses)/6,grad_norm=norm.item(),entries=entries,total_policy_tokens=total_policy,total_context_tokens=total_context,elapsed_seconds=time.monotonic()-start_time,peak_cuda_allocated=torch.cuda.max_memory_allocated())
        with (args.out/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        print('UPDATE',args.arm,step+1,record['loss'],flush=True)
    opt.zero_grad(set_to_none=True);del opt
    model.to(dtype=torch.bfloat16);model.config.use_cache=True;model.gradient_checkpointing_disable()
    checkpoint_dir=args.out/'checkpoint';model.save_pretrained(checkpoint_dir,safe_serialization=True);tok.save_pretrained(checkpoint_dir)
    save(args.out/'status.json',dict(state='complete',arm=args.arm,steps=8,checkpoint=str(checkpoint_dir.resolve()),total_policy_tokens=total_policy,total_context_tokens=total_context,seconds=time.monotonic()-start_time,peak_cuda_allocated=torch.cuda.max_memory_allocated(),**saved_parameter_change()))


if __name__=='__main__':main()
