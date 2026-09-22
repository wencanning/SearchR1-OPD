"""CPU-only one-update generation pilot. Never writes model checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/experiments/action_calibration_cpu'))
from common import M,BASE,setup,load_model,rows,save,append,sha
sp=importlib.util.spec_from_file_location('strict_role_parser',ROOT/'scripts/experiments/strict_role_cpu/local_update.py')
R=importlib.util.module_from_spec(sp);sp.loader.exec_module(R)
OLD=ROOT/'reports/action_calibration_cpu_20260916'
OUT=ROOT/'reports/er_generation_cpu_20260916'


def parse(text,mode):
    if mode=='forced_answer':
        match=re.fullmatch(r'\s*(.*?)</answer>\s*',text,re.S)
        action='answer'
    else:
        match=re.fullmatch(r'\s*(?:<think>.*?</think>\s*)?<(answer|search)>(.*?)</\1>\s*',text,re.S)
        action=match[1] if match else 'invalid'
    content=match[1 if mode=='forced_answer' else 2].strip() if match else None
    if not content or re.search(r'</?(?:think|answer|search|information)>',content):return 'invalid',None
    return action,content


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('panel',choices=['pilot','full']);args=p.parse_args()
    out=OUT/args.panel;out.mkdir(exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=M.read(OUT/'protocol.frozen.json')
    for path,h in protocol['source_sha256'].items():assert sha(ROOT/path)==h,('Frozen input/code changed',path)
    m=M.read(OLD/'inputs.frozen.json');cfg=m['update'];torch=setup()
    from transformers import AutoTokenizer,GenerationConfig,LogitsProcessor,StoppingCriteria
    from verl.trainer.ppo.core_algos import compute_policy_loss
    assert M.read(OLD/'update_samples.frozen.json')['sha256']==sha(OLD/'update_samples.jsonl')
    assert M.read(OLD/'update_targets.frozen.json')['sha256']==sha(OLD/'update_targets.jsonl')
    samples=rows(OLD/'update_samples.jsonl');targets={(r['qid'],r['condition'],r['sample']):r for r in rows(OLD/'update_targets.jsonl')}
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    for c in samples:c['strict_roles']=R.roles(tok,c['response_ids'])
    qids=protocol[args.panel+'_qids'];cases={(c['qid'],c['condition']):c for c in m['cases'] if c['qid'] in qids}
    tasks=[(q,condition,mode) for q in qids for condition,mode in protocol['modes']]
    expected={(variant,*t) for variant in protocol['variants'] for t in tasks}
    existing=rows(out/'generations.jsonl');done={(r['variant'],r['qid'],r['condition'],r['mode']) for r in existing}
    assert len(done)==len(existing) and done<=expected
    finished_variants={r['variant'] for r in rows(out/'updates.jsonl')}
    baseline_results=rows(OLD/'update_effects.jsonl')+rows(ROOT/'reports/strict_role_cpu_20260916/update_effects.jsonl')
    baseline_grad={r['variant']:r['grad_norm_before_clip'] for r in baseline_results}
    model=load_model(BASE);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    weights=list(BASE.glob('*.safetensors'));weight_meta={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in weights}
    count=sum(len(c['response_ids']) for c in samples)
    def logps(c):
        ys=c['response_ids'];x=torch.tensor([c['prefix_ids']+ys[:-1]])
        z=model(x,attention_mask=torch.ones_like(x),use_cache=False,num_logits_to_keep=len(ys)).logits[0,:,:m['vocab_size']].float()
        return z.log_softmax(-1)[torch.arange(len(ys)),torch.tensor(ys)]
    max_error=0.
    with torch.no_grad():
        for c in samples:
            lp=logps(c);max_error=max(max_error,float((lp-torch.tensor(c['old_log_probs'])).abs().max()))
            c['old_recomputed']=lp.tolist()
    assert max_error<1e-3
    save(out/'PARITY.json',dict(max_old_logprob_error=max_error,cuda_initialized=False))
    class Vocab(LogitsProcessor):
        def __call__(self,ids,scores):scores[:,m['vocab_size']:]=-float('inf');return scores
    for variant in protocol['variants']:
        if all((variant,*t) in done for t in tasks) and variant in finished_variants:continue
        model.load_state_dict(initial);model.eval();tic=time.time()
        save(out/'status.json',dict(state='running',stage='update',variant=variant,completed_generations=len(done),total_generations=len(expected),cuda_initialized=False))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        opt.zero_grad(set_to_none=True);loss_value=0.
        for c in samples:
            lp=logps(c);old=torch.tensor(c['old_recomputed']);t=targets[c['qid'],c['condition'],c['sample']]
            obs=torch.tensor(t['observed_log_probs']);er=torch.tensor(t['er_log_probs'])
            active=torch.ones_like(old) if variant=='full_er' else torch.tensor([float(r=='body') for r in c['strict_roles']]) if variant=='strict_body_er' else torch.zeros_like(old)
            advantage=cfg['lambda_distill']*(obs-old+active*(er-obs))
            loss,*_=compute_policy_loss(old[None],lp[None],advantage[None],torch.ones_like(lp)[None],.2,.2,.28,3.)
            weighted=loss*(len(c['response_ids'])/count);weighted.backward();loss_value+=float(weighted.detach())
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']))
        assert abs(grad-baseline_grad[variant])<1e-5,('Update failed to reproduce',variant,grad,baseline_grad[variant])
        opt.step();opt.zero_grad(set_to_none=True);del opt;gc.collect()
        if variant not in finished_variants:
            append(out/'updates.jsonl',dict(variant=variant,grad_norm=grad,loss=loss_value,seconds=time.time()-tic,
                common_grpo_advantage=0,checkpoint_written=False,cuda_initialized=False))
            finished_variants.add(variant)
        print('UPDATE_REPRODUCED',variant,grad,flush=True)
        for qid,condition,mode in tasks:
            key=(variant,qid,condition,mode)
            if key in done:continue
            c=cases[qid,condition];ids=c['ids']+(m['tags']['answer'] if mode=='forced_answer' else [])
            assert len(ids)+protocol['max_new_tokens']<=4096
            width=len(ids)
            class Stop(StoppingCriteria):
                def __call__(self,x,scores,**kw):
                    text=tok.decode(x[0,width:],skip_special_tokens=False)
                    return torch.tensor(['</answer>' in text or mode=='free_action' and '</search>' in text],device='cpu')
            x=torch.tensor([ids]);tic=time.time()
            config=GenerationConfig(do_sample=False,use_cache=True,max_new_tokens=protocol['max_new_tokens'],
                eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)
            with torch.no_grad():ys=model.generate(x,attention_mask=torch.ones_like(x),generation_config=config,
                stopping_criteria=[Stop()],logits_processor=[Vocab()])[0,width:].tolist()
            text=tok.decode(ys[:-1] if ys and ys[-1]==tok.eos_token_id else ys,skip_special_tokens=False)
            action,content=parse(text,mode);answer=content if action=='answer' else None
            assert not torch.cuda.is_initialized()
            append(out/'generations.jsonl',dict(variant=variant,qid=qid,condition=condition,mode=mode,
                action=action,content=content,raw_text=text,token_ids=ys,gold=c['gold'],
                answer_em=M.exact(answer,c['gold']),generated_tokens=len(ys),
                stop_reason='closed_action' if action!='invalid' else 'invalid_or_incomplete',
                seconds=time.time()-tic,cuda_initialized=False,
                interpretation='first_action_only_no_retrieval; answer_em is NOT final search-task EM'))
            done.add(key);save(out/'status.json',dict(state='running',stage='generation',variant=variant,
                completed_generations=len(done),total_generations=len(expected),cuda_initialized=False))
            print('GENERATION',len(done),len(expected),*key,action,flush=True)
    assert done==expected
    for path,meta in weight_meta.items():assert (Path(path).stat().st_size,Path(path).stat().st_mtime_ns)==meta
    save(out/'status.json',dict(state='complete',generations=len(done),variants=3,cuda_initialized=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if len(sys.argv)>1 and sys.argv[1] in ('pilot','full'):
            out=OUT/sys.argv[1];out.mkdir(exist_ok=True);save(out/'failure.json',dict(error=repr(error),state='failed'))
        raise
