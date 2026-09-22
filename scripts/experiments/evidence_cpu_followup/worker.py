"""Bounded CPU-only diagnostics. Never calls CUDA or the retrieval service."""
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
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/evidence_cpu_followup_20260915'
SOURCE=ROOT/'reports/evidence_stopping_gpu5_20260915'
sp=importlib.util.spec_from_file_location('cpu_helpers',ROOT/'scripts/experiments/evidence_use/experiment.py')
M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
TEACHER='/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'


def rows(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def append(path,r):
    with path.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()


def make_boundary(tok,record,tag):
    text=record['generation']['raw_text'];match=re.search(r'<(?:answer|search)>',text)
    if match is None:return None
    thought=text[:match.start()]
    if '<think>' in thought and '</think>' not in thought:return None
    ids=record['generation']['token_ids']
    cut=next((i for i in range(len(ids)+1) if tok.decode(ids[:i],skip_special_tokens=False)==thought),None)
    native=cut is not None
    thought_ids=ids[:cut] if native else tok.encode(thought,add_special_tokens=False)
    return dict(ids=record['input_ids']+thought_ids+tok.encode(tag,add_special_tokens=False),
        native_boundary=native,reasoning_present='</think>' in thought,reasoning_text=thought)


def prepare(tok):
    source_protocol=M.read(SOURCE/'protocol.frozen.json')
    inputs=rows(SOURCE/'first_actions.jsonl');index={(r['label'],r['qid'],r['condition']):r for r in inputs}
    candidates={q['qid']:q for q in M.read(SOURCE/'candidates.json')}
    reviews={r['qid']:r for r in M.read(SOURCE/'review.json')}
    student=[]
    for c in source_protocol['checkpoints']:
        for qid in source_protocol['selected_qids']:
            original=index[c['label'],qid,'both_hops'];boundary=make_boundary(tok,original,'<answer>')
            student.append(dict(label=c['label'],model=str(ROOT/c['path']),qid=qid,
                gold=candidates[qid]['gold'],aliases=reviews[qid]['aliases'],boundary=boundary))
    chosen=sorted(source_protocol['selected_qids'],key=lambda q:hashlib.sha256(('teacher-cpu-fixed-20260915|'+q).encode()).hexdigest())[:6]
    teacher=[]
    for qid in chosen:
        for condition in M.CONDITIONS:
            original=index['opd-step100',qid,condition]
            boundary=make_boundary(tok,original,'<')
            if boundary is None:raise RuntimeError(f'No common teacher action boundary: {qid} {condition}')
            ids=boundary['ids'];text=tok.decode(ids,skip_special_tokens=False)
            matches=list(re.finditer(r'<information>(.*?)</information>',text,re.S))
            assert matches
            start,end=matches[-1].span(1)
            lengths=[0]+[len(tok.decode(ids[:i],skip_special_tokens=False)) for i in range(1,len(ids)+1)]
            mask=[int(b>a and a>=start and b<=end) for a,b in zip(lengths,lengths[1:])]
            assert sum(mask)>0 and len(mask)==len(ids)
            teacher.append(dict(qid=qid,condition=condition,evidence_mask=mask,**boundary))
    manifest=dict(student_cases=student,teacher_cases=teacher,selected_teacher_qids=chosen,
        source_first_actions_sha256=hashlib.sha256((SOURCE/'first_actions.jsonl').read_bytes()).hexdigest(),
        source_review_sha256=hashlib.sha256((SOURCE/'review.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),alphas=[0,.5,1,1.5],
        device='cpu',dtype='float32',threads=4,student_max_new_tokens=96,
        teacher_interpretation='Conditional next-word answer/search support after shared < boundary, not full action probability or training causality.')
    path=OUT/'inputs.frozen.json'
    if path.exists():assert M.read(path)==manifest
    else:M.save(path,manifest)
    return manifest


def student_stage(manifest,tok,torch):
    from transformers import AutoModelForCausalLM,GenerationConfig,LogitsProcessor,StoppingCriteria
    class Vocab(LogitsProcessor):
        def __call__(self,ids,scores):scores[:,len(tok):]=-float('inf');return scores
    path=OUT/'student_answers.jsonl';done={(r['label'],r['qid']) for r in rows(path)}
    start=time.time();model=None;loaded=None
    for c in manifest['student_cases']:
        if (c['label'],c['qid']) in done:continue
        if loaded!=c['model']:
            if model is not None:del model;gc.collect()
            print('LOAD_CPU_STUDENT',c['label'],flush=True)
            model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,torch_dtype=torch.float32,
                attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cpu');loaded=c['model']
            assert all(p.device.type=='cpu' for p in model.parameters())
        tic=time.time();answer=None;text=None
        if c['boundary'] is not None:
            ids=c['boundary']['ids'];x=torch.tensor([ids],device='cpu');n=len(ids)
            class Stop(StoppingCriteria):
                def __call__(self,input_ids,scores,**kw):return '</answer>' in tok.decode(input_ids[0,n:],skip_special_tokens=False)
            cfg=GenerationConfig(do_sample=False,max_new_tokens=96,use_cache=True,
                eos_token_id=[tok.eos_token_id,tok.convert_tokens_to_ids('<|endoftext|>')],pad_token_id=tok.eos_token_id)
            with torch.inference_mode():
                result=model.generate(x,attention_mask=torch.ones_like(x),generation_config=cfg,
                    logits_processor=[Vocab()],stopping_criteria=[Stop()])
            text=tok.decode(result[0,n:],skip_special_tokens=True)
            if '</answer>' in text:
                candidate=text.split('</answer>')[0].strip()
                if not re.search(r'</?(?:answer|search|think)>',candidate):answer=candidate
        assert not torch.cuda.is_initialized()
        r=dict(label=c['label'],qid=c['qid'],available=c['boundary'] is not None,
            reasoning_present=c['boundary']['reasoning_present'] if c['boundary'] else None,
            answer=answer,raw_text=text,em=M.exact(answer,c['gold']),alias_em=M.exact(answer,c['aliases']),
            seconds=time.time()-tic,device='cpu',dtype='float32',cuda_initialized=False)
        append(path,r);done.add((c['label'],c['qid']))
        M.save(OUT/'status.json',dict(stage='student_cpu',state='running',completed=len(done),total=170,elapsed_seconds=time.time()-start))
        print('CPU_STUDENT',len(done),170,c['label'],c['qid'],round(r['seconds'],2),flush=True)
    if model is not None:del model;gc.collect()
    assert len(rows(path))==170


def teacher_stage(manifest,tok,torch):
    from transformers import AutoModelForCausalLM
    from verl.utils.evidence_residual import build_evidence_hidden_attention_mask
    path=OUT/'teacher_preferences.jsonl';done={(r['qid'],r['condition']) for r in rows(path)}
    pending=[c for c in manifest['teacher_cases'] if (c['qid'],c['condition']) not in done]
    if not pending:return
    words={w:tok.encode(w,add_special_tokens=False) for w in ['answer','search','think']}
    assert all(len(ids)==1 for ids in words.values());ix={w:ids[0] for w,ids in words.items()}
    model=AutoModelForCausalLM.from_pretrained(TEACHER,local_files_only=True,torch_dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cpu')
    assert all(p.device.type=='cpu' for p in model.parameters())
    def forward(c,mask):
        x=torch.tensor([c['ids']],device='cpu');valid=torch.ones_like(x)
        attention=valid if mask is None else build_evidence_hidden_attention_mask(valid,torch.tensor([mask]),torch.float32)
        with torch.inference_mode():
            z=model(x,attention_mask=attention,position_ids=torch.arange(len(c['ids']))[None,:],
                use_cache=False,num_logits_to_keep=1).logits[0,-1].float()
        assert torch.isfinite(z).all() and not torch.cuda.is_initialized()
        return z
    def stats(z):
        lp=z.log_softmax(-1)
        return dict(log_probs={w:float(lp[j]) for w,j in ix.items()},
            answer_search_log_odds=float(z[ix['answer']]-z[ix['search']]),
            conditional_answer_probability=float(torch.sigmoid(z[ix['answer']]-z[ix['search']])),
            entropy=float(-(lp.exp()*lp).sum()))
    start=time.time()
    print('LOAD_CPU_TEACHER_COMPLETE',flush=True)
    for i,c in enumerate(pending):
        tic=time.time();obs=forward(c,None);hid=forward(c,c['evidence_mask'])
        if i==0:
            zero=forward(c,[0]*len(c['ids']));error=float((obs-zero).abs().max())
            assert error<1e-4,error
            M.save(OUT/'TEACHER_NULL_MASK_CHECK.json',dict(passed=True,max_error=error,cuda_initialized=False))
        targets={str(a):stats(obs+a*(obs-hid)) for a in manifest['alphas']}
        target_entropy=targets['1.5']['entropy'];lo,hi=.001,1000.
        bracket=stats(obs/lo)['entropy']<=target_entropy<=stats(obs/hi)['entropy']
        temperature=None
        if bracket:
            for _ in range(40):
                mid=(lo*hi)**.5
                if stats(obs/mid)['entropy']<target_entropy:lo=mid
                else:hi=mid
            tau=(lo*hi)**.5;temperature=dict(tau=tau,**stats(obs/tau))
        r=dict(qid=c['qid'],condition=c['condition'],hidden=stats(hid),targets=targets,
            entropy_matched_observed=temperature,prefix_tokens=len(c['ids']),evidence_tokens=sum(c['evidence_mask']),
            delta_answer_search_log_odds=targets['1.5']['answer_search_log_odds']-targets['0']['answer_search_log_odds'],
            seconds=time.time()-tic,device='cpu',dtype='float32',cuda_initialized=False)
        append(path,r);done.add((c['qid'],c['condition']))
        M.save(OUT/'status.json',dict(stage='teacher_cpu',state='running',completed=len(done),total=24,elapsed_seconds=time.time()-start))
        print('CPU_TEACHER',len(done),24,c['qid'],c['condition'],round(r['seconds'],2),flush=True)
    del model;gc.collect()
    assert len(rows(path))==24


def main():
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260915)
    assert not torch.cuda.is_initialized()
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    manifest=prepare(tok)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','student','teacher','all']);args=parser.parse_args()
    if args.stage=='prepare':print('PREPARED',len(manifest['student_cases']),len(manifest['teacher_cases']));return
    if args.stage in ['student','all']:student_stage(manifest,tok,torch)
    if args.stage in ['teacher','all']:teacher_stage(manifest,tok,torch)
    M.save(OUT/('COMPLETE_'+args.stage+'.json'),dict(complete=True,cuda_initialized=torch.cuda.is_initialized()))
    if args.stage=='all':M.save(OUT/'status.json',dict(state='complete',student_records=170,teacher_records=24,cuda_initialized=False))


if __name__=='__main__':
    try:main()
    except Exception as e:
        M.save(OUT/'failure.json',dict(error=repr(e),at=time.strftime('%Y-%m-%d %H:%M:%S')));raise
