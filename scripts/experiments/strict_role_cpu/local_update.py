"""Disposable common-batch update isolating the ER distillation component."""
import argparse
import collections
import gc
import re
import time
from common import ROOT,OUT,M,BASE,TEACHER,setup,load_model,verify_inputs,path_score,rows,append,save,rank,sha


def roles(tok,ids):
    text=tok.decode(ids,skip_special_tokens=False)
    match=re.fullmatch(r'\s*<(search|answer)>(.*?)</\1>\s*',text,re.S)
    if match is None or re.search(r'</?(?:think|search|answer|information)>',match[2]):
        return ['unknown']*len(ids)
    body_start,body_end=match.span(2)
    tags=list(re.finditer(r'</?(?:search|answer)>',text))
    action=next(re.finditer(r'<(search|answer)>',text)).span(1)
    lengths=[0]+[len(tok.decode(ids[:i],skip_special_tokens=False)) for i in range(1,len(ids)+1)]
    result=[]
    for a,b in zip(lengths,lengths[1:]):
        if a>=body_start and b<=body_end and b>a: result.append('body')
        elif a<action[1] and b>action[0]: result.append('action')
        elif any(a>=m.start() and b<=m.end() for m in tags): result.append('format')
        else: result.append('unknown')
    return result


def sample(m,torch):
    from transformers import AutoTokenizer,GenerationConfig,LogitsProcessor,StoppingCriteria
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=load_model(BASE)
    cfg=m['update'];path=OUT/'update_samples.jsonl';old=rows(path);done={(r['qid'],r['condition'],r['sample']) for r in old}
    assert len(done)==len(old)
    cases=[c for c in m['cases'] if c['split']=='update' and c['condition'] in cfg['conditions']]
    class Vocab(LogitsProcessor):
        def __call__(self,ids,scores):scores[:,m['vocab_size']:]=-float('inf');return scores
    for index,c in enumerate(cases):
        for k in range(cfg['samples_per_state']):
            key=(c['qid'],c['condition'],k)
            if key in done:continue
            seed=cfg['seed']+index*cfg['samples_per_state']+k;torch.manual_seed(seed)
            n=len(c['ids'])
            class Stop(StoppingCriteria):
                def __call__(self,ids,scores,**kwargs):
                    text=tok.decode(ids[0,n:],skip_special_tokens=False)
                    return '</search>' in text or '</answer>' in text
            config=GenerationConfig(do_sample=True,temperature=1.0,top_k=0,top_p=1.0,max_new_tokens=cfg['max_new_tokens'],
                use_cache=True,eos_token_id=[tok.eos_token_id,tok.convert_tokens_to_ids('<|endoftext|>')],pad_token_id=tok.eos_token_id,
                return_dict_in_generate=True,output_scores=True)
            x=torch.tensor([c['ids']],device='cpu');tic=time.time()
            with torch.no_grad():
                output=model.generate(x,attention_mask=torch.ones_like(x),generation_config=config,
                    stopping_criteria=[Stop()],logits_processor=[Vocab()])
                ids=output.sequences[0,n:].tolist()
                lp=[float(z[0,:m['vocab_size']].float().log_softmax(-1)[y]) for z,y in zip(output.scores,ids)]
            assert ids and len(ids)==len(lp) and all(y<m['vocab_size'] for y in ids)
            text=tok.decode(ids,skip_special_tokens=False);categories=roles(tok,ids)
            row=dict(qid=c['qid'],condition=c['condition'],sample=k,seed=seed,prefix_ids=c['ids'],evidence_mask=c['evidence_mask'],
                response_ids=ids,old_log_probs=lp,roles=categories,raw_text=text,
                complete_action=bool(re.search(r'<(search|answer)>.*?</\1>',text,re.S)),seconds=time.time()-tic,
                device='cpu',dtype='float32',cuda_initialized=torch.cuda.is_initialized())
            assert not row['cuda_initialized'];append(path,row);done.add(key)
            save(OUT/'status.json',dict(state='running',stage='update_sampling',completed=len(done),total=32,cuda_initialized=False))
            print('UPDATE_SAMPLE',len(done),32,*key,len(ids),flush=True)
            del output
    assert len(rows(path))==32
    del model;gc.collect()
    save(OUT/'update_samples.frozen.json',dict(sha256=sha(path),n=32,input_sha256=sha(OUT/'inputs.frozen.json')))


def teacher(m,torch):
    from verl.utils.evidence_residual import build_evidence_hidden_attention_mask
    assert M.read(OUT/'update_samples.frozen.json')['sha256']==sha(OUT/'update_samples.jsonl')
    samples=rows(OUT/'update_samples.jsonl');path=OUT/'update_targets.jsonl';old=rows(path)
    done={(r['qid'],r['condition'],r['sample']) for r in old};assert len(done)==len(old)
    model=load_model(TEACHER)
    for c in samples:
        key=(c['qid'],c['condition'],c['sample'])
        if key in done:continue
        tic=time.time();ys=c['response_ids'];ids=c['prefix_ids']+ys[:-1];x=torch.tensor([ids],device='cpu');mask=torch.ones_like(x)
        hidden_mask=build_evidence_hidden_attention_mask(mask,torch.tensor([c['evidence_mask']+[0]*(len(ys)-1)]),torch.float32)
        def forward(attn):
            with torch.no_grad():return model(x,attention_mask=attn,use_cache=False,num_logits_to_keep=len(ys)).logits[0,:,:m['vocab_size']].float()
        obs=forward(mask);hid=forward(hidden_mask)
        if not(OUT/'UPDATE_TEACHER_NULL_CHECK.json').exists():
            zero=forward(build_evidence_hidden_attention_mask(mask,torch.zeros_like(mask),torch.float32));err=float((obs-zero).abs().max());assert err<1e-4
            save(OUT/'UPDATE_TEACHER_NULL_CHECK.json',dict(passed=True,max_error=err,cuda_initialized=False));del zero
        def selected(z):return z.log_softmax(-1)[torch.arange(len(ys)),torch.tensor(ys)].tolist()
        er=obs+m['update']['alpha']*(obs-hid)
        row=dict(qid=c['qid'],condition=c['condition'],sample=c['sample'],observed_log_probs=selected(obs),
            er_log_probs=selected(er),hidden_log_probs=selected(hid),seconds=time.time()-tic,device='cpu',cuda_initialized=False)
        assert torch.isfinite(obs).all() and torch.isfinite(hid).all() and torch.isfinite(er).all() and not torch.cuda.is_initialized()
        append(path,row);done.add(key);del obs,hid,er
        save(OUT/'status.json',dict(state='running',stage='update_teacher',completed=len(done),total=32,cuda_initialized=False))
        print('UPDATE_TARGET',len(done),32,*key,flush=True)
    assert len(rows(path))==32
    del model;gc.collect()
    save(OUT/'update_targets.frozen.json',dict(sha256=sha(path),sample_sha256=sha(OUT/'update_samples.jsonl')))


def update(m,torch):
    from transformers import AutoTokenizer
    from verl.trainer.ppo.core_algos import compute_policy_loss
    samples=rows(OUT/'update_samples.jsonl');targets=rows(OUT/'update_targets.jsonl')
    assert M.read(OUT/'update_targets.frozen.json')['sha256']==sha(OUT/'update_targets.jsonl')
    assert M.read(OUT/'update_samples.frozen.json')['sha256']==sha(OUT/'update_samples.jsonl')
    ti={(c['qid'],c['condition'],c['sample']):c for c in targets};assert len(ti)==32
    cfg=m['update'];tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=load_model(BASE)
    for c in samples:c['roles']=roles(tok,c['response_ids'])
    probe_qids=sorted(m['split_qids']['test'],key=lambda q:rank(q,'update-probes-20260916'))[:8]
    probes=[c for c in m['cases'] if c['split']=='test' and c['qid'] in probe_qids]
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    token_count=sum(len(c['response_ids']) for c in samples)
    def logps(c):
        ys=c['response_ids'];x=torch.tensor([c['prefix_ids']+ys[:-1]],device='cpu')
        z=model(x,attention_mask=torch.ones_like(x),use_cache=False,num_logits_to_keep=len(ys)).logits[0,:,:m['vocab_size']].float()
        return z.log_softmax(-1)[torch.arange(len(ys)),torch.tensor(ys)]
    def probe():
        output=[]
        for c in probes:
            scores={a:path_score(model,c['ids'],tag,m['vocab_size'],torch) for a,tag in m['tags'].items()}
            row=dict(qid=c['qid'],condition=c['condition'],log_odds=scores['answer']['log_probability']-scores['search']['log_probability'])
            if c['condition']=='both_hops':
                body=tok.encode(' '+c['gold'][0],add_special_tokens=False);assert body and len(body)<128
                scored=path_score(model,c['ids']+m['tags']['answer'],body,m['vocab_size'],torch)
                row['reference_body_mean_logp']=scored['log_probability']/len(body)
            output.append(row)
        return output
    maximum_error=0.;signal=[]
    with torch.no_grad():
        for c in samples:
            old=logps(c);saved=torch.tensor(c['old_log_probs']);error=float((old-saved).abs().max());maximum_error=max(maximum_error,error)
            assert error<1e-3,('CPU cached generation / no-cache scoring mismatch',error)
            c['recomputed_old_log_probs']=old.tolist()
            t=ti[c['qid'],c['condition'],c['sample']]
            for i,role in enumerate(c['roles']):
                base=cfg['lambda_distill']*(t['observed_log_probs'][i]-float(old[i]))
                delta=cfg['lambda_distill']*(t['er_log_probs'][i]-t['observed_log_probs'][i])
                signal.append(dict(qid=c['qid'],condition=c['condition'],sample=c['sample'],token_index=i,token_id=c['response_ids'][i],
                    role=role,opd_advantage=base,er_increment=delta,full_er_advantage=base+delta))
    save(OUT/'UPDATE_STUDENT_PARITY.json',dict(passed=True,max_error=maximum_error,cuda_initialized=False))
    save(OUT/'update_token_signals.json',signal)
    before=probe();save(OUT/'update_probe_before.json',before)
    done={r['variant'] for r in rows(OUT/'update_effects.jsonl')}
    for variant in cfg['variants']:
        if variant in done:continue
        tic=time.time();model.load_state_dict(initial);model.eval()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.0)
        opt.zero_grad(set_to_none=True);loss_value=0.
        for c in samples:
            lp=logps(c);old=torch.tensor(c['recomputed_old_log_probs']);t=ti[c['qid'],c['condition'],c['sample']]
            obs=torch.tensor(t['observed_log_probs']);er=torch.tensor(t['er_log_probs'])
            if variant=='full_er':active=torch.ones_like(old)
            elif variant=='action_only_er':active=torch.tensor([float(role in ['action','action_mixed']) for role in c['roles']])
            elif variant=='strict_body_er':active=torch.tensor([float(role=='body') for role in c['roles']])
            elif variant=='format_only_er':active=torch.tensor([float(role=='format') for role in c['roles']])
            else:active=torch.zeros_like(old)
            advantage=cfg['lambda_distill']*(obs-old+active*(er-obs))
            loss,*_=compute_policy_loss(old[None,:],lp[None,:],advantage[None,:],torch.ones_like(lp)[None,:],.2,.2,.28,3.)
            weighted=loss*(len(c['response_ids'])/token_count);weighted.backward();loss_value+=float(weighted.detach())
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']));assert torch.isfinite(torch.tensor(grad))
        opt.step();opt.zero_grad(set_to_none=True);after=probe()
        assert not torch.cuda.is_initialized()
        append(OUT/'update_effects.jsonl',dict(variant=variant,before=before,after=after,grad_norm_before_clip=grad,loss=loss_value,
            seconds=time.time()-tic,fresh_optimizer=True,common_grpo_advantage=0.0,total_tokens=token_count,
            device='cpu',dtype='float32',cuda_initialized=False,no_checkpoint_written=True))
        done.add(variant);save(OUT/'status.json',dict(state='running',stage='local_update',completed=len(done),total=len(cfg['variants']),cuda_initialized=False))
        print('LOCAL_UPDATE_COMPLETE',variant,'grad_norm',grad,'seconds',round(time.time()-tic,1),flush=True)
        del opt;gc.collect()
    assert len(rows(OUT/'update_effects.jsonl'))==len(cfg['variants'])
    del model,initial;gc.collect()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['sample','teacher','update']);args=parser.parse_args()
    torch=setup();m=verify_inputs()
    {'sample':sample,'teacher':teacher,'update':update}[args.stage](m,torch)


if __name__=='__main__':
    try:main()
    except Exception as e:save(OUT/'failure.json',dict(stage='local_update',error=repr(e)));raise
