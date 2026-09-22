"""One visible GPU per read-only HF worker. No training or live-checkpoint writes."""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

from common import ROOT, TARGETS, append, checkpoint, config, exact, f1, normalize, read, rows
sys.path.insert(0, str(ROOT))


def entropy(z):
    lp = z.log_softmax(-1)
    return -(lp.exp()*lp).sum(-1)


def matched(obs, er):
    import torch
    target = entropy(er); centered = obs-obs.max(-1, keepdim=True).values
    lo = torch.full_like(target, -5.0); hi = torch.full_like(target, 5.0)
    for _ in range(12):
        low_bad = entropy(centered/lo.exp()[:,None]) > target+1e-5
        high_bad = entropy(centered/hi.exp()[:,None]) < target-1e-5
        if not (low_bad.any() or high_bad.any()): break
        lo = torch.where(low_bad,lo-2.3,lo); hi = torch.where(high_bad,hi+2.3,hi)
    if (entropy(centered/lo.exp()[:,None]) > target+1e-4).any() or (entropy(centered/hi.exp()[:,None]) < target-1e-4).any():
        raise ValueError('Entropy target cannot be bracketed (possibly tied maxima)')
    for _ in range(32):
        mid = (lo+hi)/2; low = entropy(centered/mid.exp()[:,None]) < target
        lo = torch.where(low,mid,lo); hi = torch.where(low,hi,mid)
    tau = ((lo+hi)/2).exp(); z = centered/tau[:,None]
    gap = float((entropy(z)-target).abs().max())
    if gap > 1e-4: raise ValueError(f'Entropy mismatch {gap}')
    return z, gap


class Teacher:
    def __init__(self, c):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.c = c; self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(c['teacher'],local_files_only=True)
        student_tok = AutoTokenizer.from_pretrained(c['student'],local_files_only=True)
        if self.tok.get_vocab() != student_tok.get_vocab(): raise ValueError('Teacher/student token IDs differ')
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = AutoModelForCausalLM.from_pretrained(c['teacher'],local_files_only=True,
             torch_dtype=getattr(torch,c['teacher_dtype']),attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
        self.vocab = len(self.tok)

    def forward(self, ids, evidence, hidden=False, n=1, cache=None):
        from verl.utils.evidence_residual import build_evidence_hidden_attention_mask
        t = self.torch; device = self.model.device
        if len(ids) != len(evidence): raise ValueError('Mask/token length mismatch')
        x = t.tensor([ids if cache is None else ids[-1:]],device=device)
        mask = t.ones((1,len(ids)),device=device,dtype=t.long)
        if hidden:
            if cache is None:
                mask = build_evidence_hidden_attention_mask(mask,t.tensor([evidence],device=device),self.model.dtype)
            else:
                mask = t.zeros((1,1,1,len(ids)),device=device,dtype=self.model.dtype)
                mask.masked_fill_(t.tensor(evidence,device=device,dtype=t.bool)[None,None,None,:],t.finfo(self.model.dtype).min)
        pos = t.arange(len(ids)-x.shape[1],len(ids),device=device)[None,:]
        with t.inference_mode():
            result = self.model(input_ids=x,attention_mask=mask,position_ids=pos,past_key_values=cache,
                                use_cache=True,num_logits_to_keep=n,return_dict=True)
        z = result.logits[0,-n:,:self.vocab].float()
        if not t.isfinite(z).all(): raise FloatingPointError('Non-finite teacher logits')
        return z, result.past_key_values

    def witness(self, ids, mask):
        """Check zero-mask and independently recomputed cache continuation before use."""
        z,_ = self.forward(ids,mask)
        zero,_ = self.forward(ids,[0]*len(mask),True)
        errors = {'zero_mask': float((z-zero).abs().max())}
        for hidden in (False,True):
            a,cache = self.forward(ids,mask,hidden)
            token = int(a.argmax(-1)[0]); ext = ids+[token]; em = mask+[0]
            cached,_ = self.forward(ext,em,hidden,cache=cache)
            plain,_ = self.forward(ext,em,hidden)
            errors['cache_hidden' if hidden else 'cache_observed'] = float((cached-plain).abs().max())
        tolerance = 0.003 if self.c['teacher_dtype']=='float32' else 0.15
        if max(errors.values()) > tolerance: raise ValueError(f'Precision witness failed: {errors}')
        return errors

    def generate(self, prefix, mask, target):
        ids = list(prefix); em = list(mask); new = []; obs_cache = hid_cache = None
        gaps = []; reason = 'max_tokens'
        for _ in range(self.c['max_answer_tokens']):
            if len(ids)>=self.c['max_context']: reason='context_limit'; break
            obs,obs_cache = self.forward(ids,em,cache=obs_cache)
            z = obs
            if target != 'observed':
                hid,hid_cache = self.forward(ids,em,True,cache=hid_cache)
                if target=='hidden': z=hid
                else:
                    er = obs+self.c['alpha']*(obs-hid)
                    if target=='er': z=er
                    else: z,gap=matched(obs,er); gaps.append(gap)
            token = int(z.argmax(-1)[0]); new.append(token); ids.append(token); em.append(0)
            text = self.tok.decode(new,skip_special_tokens=False)
            if '</answer>' in text: reason='answer_closed'; break
            if token == self.tok.eos_token_id: reason='eos'; break
        text = self.tok.decode(new,skip_special_tokens=False)
        answer = text.split('</answer>')[0].strip() if reason=='answer_closed' else None
        return dict(text=text,answer=answer,stop_reason=reason,token_ids=new,max_entropy_error=max(gaps,default=0))

    def score_alias(self, prefix, mask, answer, leading):
        t = self.torch
        labels = self.tok.encode(leading+answer+'</answer>',add_special_tokens=False)
        if len(prefix)+len(labels)>self.c['max_context']: raise ValueError('Candidate exceeds context budget')
        ids = prefix+labels[:-1]; em = mask+[0]*(len(labels)-1)
        obs,_ = self.forward(ids,em,n=len(labels)); hid,_ = self.forward(ids,em,True,n=len(labels))
        er = obs+self.c['alpha']*(obs-hid); temp,gap = matched(obs,er)
        ix = t.tensor(labels,device=obs.device)[:,None]
        result = {}
        for name,z in zip(TARGETS,(hid,obs,er,temp)):
            lp = z.log_softmax(-1).gather(-1,ix).squeeze(-1)
            result[name] = dict(total=float(lp.sum()),mean=float(lp.mean()),token_logps=lp.tolist())
        return dict(answer=answer,token_ids=labels,targets=result,max_entropy_error=gap)


def observation(tok, text, budget):
    wrapped = '\n\n<information>'+text.strip()+'</information>\n\n'
    e = tok(wrapped,add_special_tokens=False,return_offsets_mapping=True)
    start = wrapped.index('<information>')+len('<information>'); end=wrapped.index('</information>')
    # A BPE token may contain both body punctuation and `</`. Keep such tokens
    # visible and retained: masking/dropping it would also hide/corrupt the tag.
    mask = [int(b>a and a>=start and b<=end) for a,b in e['offset_mapping']]
    fixed = [i for i,m in enumerate(mask) if not m]
    if len(fixed)>budget: raise ValueError('Observation budget cannot retain tags')
    keep = set(fixed+[i for i,m in enumerate(mask) if m][:budget-len(fixed)])
    return [v for i,v in enumerate(e['input_ids']) if i in keep],[v for i,v in enumerate(mask) if i in keep]


def retrieve(c, query):
    import requests
    r=requests.post(c['retriever'],json={'queries':[query],'topk':3,'return_scores':True},timeout=120)
    r.raise_for_status(); result=r.json()['result'][0]
    return ''.join(f"Doc {i+1}(Title: {d['document']['contents'].split(chr(10))[0]}) "+
                   '\n'.join(d['document']['contents'].split('\n')[1:])+'\n' for i,d in enumerate(result))


def rollout(model, tok, c, q):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
    class Stop(StoppingCriteria):
        def __init__(self,n): self.n=n
        def __call__(self,input_ids,scores,**kw):
            s=tok.decode(input_ids[0,self.n:],skip_special_tokens=False)
            return '</search>' in s or '</answer>' in s
    class Vocab(LogitsProcessor):
        def __call__(self,input_ids,scores): scores[:,len(tok):]=-float('inf'); return scores
    ids=tok.apply_chat_template(q['prompt'],add_generation_prompt=True,tokenize=True)
    mask=[0]*len(ids); obs=[]; answer=None; native=None; reason='max_searches'; searches=0
    for _ in range(c['max_searches']+1):
        budget=min(c['max_action_tokens'],c['max_context']-len(ids))
        if budget<1: reason='context_limit'; break
        x=torch.tensor([ids],device=model.device)
        with torch.inference_mode():
            out=model.generate(x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=budget,
                stopping_criteria=StoppingCriteriaList([Stop(len(ids))]),
                logits_processor=LogitsProcessorList([Vocab()]),pad_token_id=tok.eos_token_id)
        action=out[0,len(ids):].tolist(); text=tok.decode(action,skip_special_tokens=False)
        # Locate a token-exact native opening boundary without retokenizing policy text.
        found=re.search(r'<answer>(.*?)</answer>',text,re.S)
        search=re.search(r'<search>(.*?)</search>',text,re.S)
        if found and (not search or found.start()<search.start()):
            answer=found[1].strip(); boundary=found.start(1)
            cuts=[i for i in range(1,len(action)+1) if len(tok.decode(action[:i],skip_special_tokens=False))==boundary]
            if cuts:
                cut=cuts[0]; native=dict(prefix_ids=ids+action[:cut],evidence_mask=mask+[0]*cut,leading=' ')
            reason='answer_closed'; ids+=action; mask += [0]*len(action); break
        ids+=action; mask += [0]*len(action)
        if search:
            if searches>=c['max_searches']: reason='max_searches'; break
            body=retrieve(c,search[1].strip()); oi,om=observation(tok,body,c['max_observation_tokens'])
            if len(ids)+len(oi)>=c['max_context']: reason='context_limit'; break
            ids+=oi; mask+=om; searches+=1
            obs.append(dict(query=search[1].strip(),visible_observation=tok.decode(oi),token_ids=oi,evidence_mask=om))
        else: reason='invalid_or_incomplete_action'; break
    return dict(answer=answer,reference_em=exact(answer,q['gold']),reference_f1=f1(answer,q['gold']),
                native=native,observations=obs,searches=searches,stop_reason=reason,
                trajectory_ids=ids,trajectory_text=tok.decode(ids,skip_special_tokens=False))


def collect(c,args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    panel=read(Path(c['output'])/'panels.json')['evaluation' if args.stage=='evaluate' else 'diagnostic']
    if args.limit: panel=panel[:args.limit]
    panel=panel[args.rank::2]
    out=Path(c['output'])/args.label/f'rollouts-{args.rank}.jsonl'
    done={(r['method'],r['seed'],r['step'],r['qid']) for r in rows(out)}
    tok=AutoTokenizer.from_pretrained(c['student'],local_files_only=True)
    for step in args.steps:
        todo=[q for q in panel if (args.method,args.seed,step,q['qid']) not in done]
        if not todo: continue
        path=checkpoint(c,args.method,args.seed,step)
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,
                  attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
        for q in todo:
            t=time.time(); result=rollout(model,tok,c,q)
            append(out,dict(qid=q['qid'],question=q['question'],gold=q['gold'],method=args.method,
                   seed=args.seed,step=step,checkpoint=path,seconds=time.time()-t,**result))
            print('ROLLOUT',args.label,args.rank,step,q['qid'],result['stop_reason'],flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    if args.stage=='evaluate': return
    # Teacher candidates are collected before annotations and ER scoring.
    teacher=Teacher(c); dest=out.with_name(f'candidates-{args.rank}.jsonl')
    done={r['key'] for r in rows(dest)}
    for r in rows(out):
        key=f"{r['method']}:{r['seed']}:{r['step']}:{r['qid']}"
        if key in done: continue
        record=dict(key=key,**r)
        if r['native'] and any(r['native']['evidence_mask']):
            p=r['native']; record['witness']=teacher.witness(p['prefix_ids'],p['evidence_mask'])
            record['teacher_generations']={t:teacher.generate(p['prefix_ids'],p['evidence_mask'],t) for t in ('observed','hidden')}
        elif r['answer'] is None: record['technical_exclusion']='no_complete_answer'
        elif not r['native']: record['technical_exclusion']='no_exact_answer_boundary'
        else: record['technical_exclusion']='no_retrieved_evidence'
        append(dest,record); print('CANDIDATES',key,flush=True)


def validate_annotation(a):
    if a.get('reviewed') is not True or not isinstance(a.get('sufficient'),bool) or not a.get('reviewer'):
        raise ValueError(f"Incomplete blind review: {a.get('key')}")
    if not isinstance(a.get('evidence_reason'),str) or not a['evidence_reason'].strip():
        raise ValueError('Explain evidence sufficiency or exclusion using visible observations')
    if not a['sufficient']: return
    groups=[a.get('correct_aliases',[])]+a.get('wrong_classes',[])
    if len(groups)<2 or any(not isinstance(g,list) or not g or any(not isinstance(s,str) or not s.strip() for s in g) for g in groups):
        raise ValueError(f"Need correct aliases and at least one frozen wrong answer class: {a['key']}")
    flat=[s for group in groups for s in group]
    if len(set(flat)) != len(flat): raise ValueError('Duplicate or cross-class alias')
    seen=set()
    for group in groups:
        normalized={normalize(s) for s in group}
        if seen & normalized: raise ValueError('Normalized aliases overlap across answer classes')
        seen |= normalized


def score(c,args):
    anns=read(Path(c['output'])/args.label/'annotations.json')
    for a in anns: validate_annotation(a)
    bykey={a['key']:a for a in anns}
    if len(bykey)!=len(anns): raise ValueError('Duplicate annotation keys')
    source=rows(Path(c['output'])/args.label/f'candidates-{args.rank}.jsonl')
    expected={r['key'] for r in source}
    if not expected<=bykey.keys(): raise ValueError('Review must cover every sampled state, including exclusions')
    dest=Path(c['output'])/args.label/f'scores-{args.rank}.jsonl'; done={r['key'] for r in rows(dest)}
    teacher=Teacher(c)
    for r in source:
        if r['key'] in done: continue
        a=bykey[r['key']]; result=dict(key=r['key'],qid=r['qid'],step=r['step'],annotation=a)
        if r.get('technical_exclusion') or not a['sufficient']:
            result['excluded']=r.get('technical_exclusion') or 'evidence_not_sufficient'
        else:
            p=r['native']; result['witness']=teacher.witness(p['prefix_ids'],p['evidence_mask'])
            groups=[a['correct_aliases']]+a['wrong_classes']; scored=[]
            for aliases in groups:
                scored.append([teacher.score_alias(p['prefix_ids'],p['evidence_mask'],s,p['leading']) for s in aliases])
            result['alias_scores']=scored; margins={}
            import torch
            for target in TARGETS:
                masses=[float(torch.logsumexp(torch.tensor([s['targets'][target]['total'] for s in group],dtype=torch.float64),0)) for group in scored]
                margins[target]=masses[0]-max(masses[1:])
            result['margins']=margins
            result['generations']=dict(r['teacher_generations'])
            for target in ('er','temperature'):
                result['generations'][target]=teacher.generate(p['prefix_ids'],p['evidence_mask'],target)
            for gen in result['generations'].values():
                gen['reference_em']=exact(gen['answer'],r['gold'])
                gen['reviewed_alias_em']=exact(gen['answer'],a['correct_aliases'])
        append(dest,result); print('SCORED',r['key'],flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['collect','evaluate','score'],required=True)
    p.add_argument('--label',required=True); p.add_argument('--method',default='opd')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--rank',type=int,required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[0]); p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(); c=config(a.config)
    import torch
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    if torch.cuda.device_count()!=1: raise RuntimeError('Worker requires exactly one visible GPU')
    if a.stage=='score': score(c,a)
    else: collect(c,a)

if __name__=='__main__': main()
