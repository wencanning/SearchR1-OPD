"""Follow-up: answer intervention and complete controlled/natural rollouts.

Frozen fresh panels, single GPU, no teacher calls or training. Controlled normal
rollouts and forced answers share each model's first generated reasoning text.
Natural rollouts start from the question alone. All stopping failures remain in
the denominator. Never treat an unfinished trajectory as a correct answer.
"""
import argparse
import collections
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/evidence_stopping_gpu5_20260915'
SPEC=importlib.util.spec_from_file_location('pilot_helpers',ROOT/'scripts/experiments/evidence_use/experiment.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
GPU=M.GPU
CONDITIONS=M.CONDITIONS
MAX_CONTEXT=5120
MAX_ACTION=512
MAX_FORCED=128
BATCH=8


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush()


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Retriever:
    def __init__(self):
        import requests
        self.session=requests.Session();self.session.trust_env=False
        self.path=OUT/'retrieval_cache.json';self.cache=M.read(self.path) if self.path.exists() else {}
    def search(self,query):
        if query not in self.cache:
            # HTTP errors are technical failures, never empty-evidence successes.
            r=self.session.post('http://127.0.0.1:8000/retrieve',
                json=dict(queries=[query],topk=3,return_scores=True),timeout=120)
            r.raise_for_status();self.cache[query]=r.json()['result'][0];M.save(self.path,self.cache)
        return self.cache[query]


def observation(tok, docs):
    # Keep protocol delimiters intact. Truncate only retrieval BODY to 512 tokens.
    text='\n'.join(f"Doc {i+1}(Title: {d['document']['contents'].split(chr(10))[0].strip()}) "+
        '\n'.join(d['document']['contents'].split('\n')[1:]) for i,d in enumerate(docs))
    body=tok.encode(text,add_special_tokens=False)
    return (tok.encode('\n\n<information>',add_special_tokens=False)+body[:512]+
            tok.encode('</information>\n\n',add_special_tokens=False)), len(body)>512


def generate(model,tok,prompts,forced=False):
    import torch
    from transformers import GenerationConfig,LogitsProcessor,StoppingCriteria
    width=max(map(len,prompts));budget=min(MAX_FORCED if forced else MAX_ACTION,MAX_CONTEXT-width)
    if budget<1:raise RuntimeError('Generation called with no context budget')
    class Vocab(LogitsProcessor):
        def __call__(self,input_ids,scores):scores[:,len(tok):]=-float('inf');return scores
    class Stop(StoppingCriteria):
        def __call__(self,input_ids,scores,**kw):
            text=tok.batch_decode(input_ids[:,width:],skip_special_tokens=False)
            return torch.tensor(['</answer>' in s if forced else '</search>' in s or '</answer>' in s for s in text],device=input_ids.device)
    x=torch.tensor([[tok.pad_token_id]*(width-len(p))+p for p in prompts],device='cuda')
    mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device='cuda')
    cfg=GenerationConfig(do_sample=False,max_new_tokens=budget,use_cache=True,
        eos_token_id=[tok.eos_token_id,tok.convert_tokens_to_ids('<|endoftext|>')],pad_token_id=tok.pad_token_id)
    with torch.inference_mode():
        output=model.generate(x,attention_mask=mask,generation_config=cfg,
            stopping_criteria=[Stop()],logits_processor=[Vocab()])[:,width:]
    result=[]
    for ids in output.tolist():
        # Remove batch padding after termination, preserve generated prefix tokens.
        eos=next((i for i,v in enumerate(ids) if v in cfg.eos_token_id),len(ids))
        ids=ids[:eos]
        result.append(dict(token_ids=ids,raw_text=tok.decode(ids,skip_special_tokens=False),
            new_tokens=len(ids),budget=budget,ended_by_eos=eos<len(output[0])))
    return result


def force_prefix(tok,input_ids,g):
    """Preserve generated pre-action reasoning; force only the action opening.

    Native token prefix is retained if its boundary is exact. Otherwise the
    preserved text is retokenized and this is explicitly recorded. Incomplete
    reasoning is not silently completed. Already-answering cases provide a
    replay control for any decoding/token-boundary effect.
    """
    text=g['raw_text'];opening=re.search(r'<(?:search|answer)>',text)
    if opening is None or '</think>' not in text[:opening.start()]:return None
    prefix=text[:opening.start()]
    cuts=[i for i in range(len(g['token_ids'])+1)
          if tok.decode(g['token_ids'][:i],skip_special_tokens=False)==prefix]
    native=bool(cuts)
    thought_ids=g['token_ids'][:cuts[0]] if native else tok.encode(prefix,add_special_tokens=False)
    return dict(ids=input_ids+thought_ids+tok.encode('<answer>',add_special_tokens=False),
                reasoning_text=prefix,native_reasoning_boundary=native)


def forced_answer(g):
    if '</answer>' not in g['raw_text']:return None
    s=g['raw_text'].split('</answer>')[0].strip()
    # Do not score a nested tool/answer/reasoning protocol as a direct answer.
    return None if re.search(r'</?(?:search|think|answer)>',s) else s


def make_result(state, q):
    first=state['actions'][0] if state['actions'] else None
    answer=state['answer']
    return dict(label=state['label'],qid=q['qid'],condition=state['condition'],
        final_answer=answer,final_em=M.exact(answer,q['gold']),
        final_alias_em=M.exact(answer,q.get('aliases',q['gold'])),stop_reason=state['stop'],
        search_calls=state['calls'],search_attempts=state['attempts'],
        total_generated_tokens=sum(a['generation']['new_tokens'] for a in state['actions']),
        first_action=first['action'] if first else None,
        immediate_answer=first['content'] if first and first['action']=='answer' else None,
        immediate_em=M.exact(first['content'] if first and first['action']=='answer' else None,q['gold']),
        immediate_alias_em=M.exact(first['content'] if first and first['action']=='answer' else None,q.get('aliases',q['gold'])),
        actions=state['actions'],final_context_ids=state['ids'])


def trajectory_batch(model,tok,retriever,tasks,label,first_store):
    states=[]
    for task in tasks:
        q=task['q'];cond=task['condition']
        ids=tok.apply_chat_template(q['prompt'],tokenize=True,add_generation_prompt=True) if cond=='natural' else M.make_prefix(tok,q,cond)[0]
        states.append(dict(q=q,label=label,condition=cond,ids=ids,initial_ids=list(ids),actions=[],
            calls=0,attempts=0,answer=None,stop=None,limit=4 if cond=='natural' else 3))
    while any(s['stop'] is None for s in states):
        active=[]
        for s in states:
            if s['stop'] is not None:continue
            # Fixed full action budget avoids longest-prompt-dependent truncation.
            if len(s['ids'])+MAX_ACTION>MAX_CONTEXT:s['stop']='context_limit'
            else:active.append(s)
        if not active:break
        generated=generate(model,tok,[s['ids'] for s in active])
        for s,g in zip(active,generated):
            action,content,text=M.parse_action(g['raw_text'])
            if action=='search' and not content:action='invalid'
            entry=dict(action=action,content=content,generation=g,context_length=len(s['ids']))
            if not s['actions']:
                record=dict(label=label,qid=s['q']['qid'],condition=s['condition'],
                    input_ids=s['initial_ids'],generation=g,action=action,content=content)
                first_store[label,s['q']['qid'],s['condition']]=record
                append(OUT/'first_actions.jsonl',record)
            s['ids']+=g['token_ids'];s['actions'].append(entry)
            if action=='answer':s['answer']=content;s['stop']='answer_closed';continue
            if action!='search':s['stop']='invalid_or_incomplete_action';continue
            s['attempts']+=1
            if s['calls']>=s['limit']:s['stop']='search_budget';continue
            docs=retriever.search(content);s['calls']+=1
            obs,truncated=observation(tok,docs)
            entry.update(retrieval_query=content,observation_ids=obs,observation_body_truncated=truncated,
                retrieved_titles=[d['document']['contents'].split('\n')[0].strip('"') for d in docs])
            if len(s['ids'])+len(obs)+MAX_ACTION>MAX_CONTEXT:s['stop']='context_limit';continue
            s['ids']+=obs
    return [make_result(s,s['q']) for s in states]


def summarize(trajectories,forced):
    groups=collections.defaultdict(list)
    for r in trajectories:groups[r['label'],r['condition']].append(r)
    summary=[]
    for (label,condition),rs in groups.items():
        n=len(rs);summary.append(dict(label=label,condition=condition,n=n,
            immediate_answer_rate=sum(r['first_action']=='answer' for r in rs)/n,
            immediate_em=sum(r['immediate_em'] for r in rs)/n,
            final_em=sum(r['final_em'] for r in rs)/n,
            final_alias_em=sum(r['final_alias_em'] for r in rs)/n,
            mean_extra_searches=sum(r['search_calls'] for r in rs)/n,
            search_budget_rate=sum(r['stop_reason']=='search_budget' for r in rs)/n,
            no_final_answer_rate=sum(r['final_answer'] is None for r in rs)/n))
    M.save(OUT/'trajectory_summary.json',summary)
    if summary:
        with (OUT/'trajectory_summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    fs=[]
    for label in sorted({r['label'] for r in forced}):
        rs=[r for r in forced if r['label']==label];n=len(rs)
        fs.append(dict(label=label,n=n,available=sum(r['available'] for r in rs),
            forced_em=sum(r['forced_em'] for r in rs)/n,
            forced_alias_em=sum(r['forced_alias_em'] for r in rs)/n,
            ordinary_immediate_em=sum(r['ordinary_immediate_em'] for r in rs)/n,
            originally_answering=sum(r['ordinary_action']=='answer' for r in rs),
            answer_replay_differences=sum(r['ordinary_action']=='answer' and r['forced_answer']!=r['ordinary_content'] for r in rs)))
    M.save(OUT/'forced_summary.json',fs)
    lines=['# GPU 5 stopping follow-up (incremental results)','',
        'Fresh fixed panels; historical runs, not independent training seeds. EM is exact reference matching.',
        'Controlled conditions allow 3 additional searches after a supplied observation. Natural allows 4 searches from the question.',
        'Forced answers reuse each model\'s first generated reasoning and replace its action opening with <answer>. Missing boundaries remain failures.',
        '', '|Model|Condition|N|Immediate EM|Final EM|Mean additional searches|No final answer|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['label']}|{r['condition']}|{r['n']}|{r['immediate_em']:.1%}|{r['final_em']:.1%}|{r['mean_extra_searches']:.2f}|{r['no_final_answer_rate']:.1%}|")
    lines+=['','|Model|N|Force available|Normal immediate EM|Forced EM|Forced alias EM|','|---|---:|---:|---:|---:|---:|']
    for r in fs:lines.append(f"|{r['label']}|{r['n']}|{r['available']}|{r['ordinary_immediate_em']:.1%}|{r['forced_em']:.1%}|{r['forced_alias_em']:.1%}|")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def inputs_and_protocol():
    panel=M.read(OUT/'candidates.json');review=M.read(OUT/'review.json');natural=M.read(OUT/'natural_panel.json')
    sel=M.read(OUT/'selection.json')
    assert M.digest(panel)==sel['candidates_sha256'] and M.digest(natural)==sel['natural_sha256']
    assert {r['qid'] for r in review}=={q['qid'] for q in panel}
    assert all(isinstance(r['include'],bool) and r['reason'] for r in review)
    rm={r['qid']:r for r in review}
    controlled=[dict(q,aliases=rm[q['qid']]['aliases']) for q in panel if rm[q['qid']]['include']]
    assert len(controlled)>=24
    assert not {q['qid'] for q in panel}&{q['qid'] for q in natural}
    checkpoints=M.read(ROOT/'reports/evidence_use_gpu5_20260915/protocol.frozen.json')['checkpoints']
    protocol=dict(checkpoints=checkpoints,selected_qids=[q['qid'] for q in controlled],
        natural_qids=[q['qid'] for q in natural],candidates_hash=M.digest(panel),review_hash=M.digest(review),
        natural_hash=M.digest(natural),conditions=CONDITIONS,max_context=MAX_CONTEXT,max_action=MAX_ACTION,
        max_forced=MAX_FORCED,batch_size=BATCH,natural_search_limit=4,controlled_additional_search_limit=3,
        forcing='preserve model-generated reasoning before first action; substitute answer opening; token-exact prefix when possible',
        primary='both_hops forced reference EM vs ordinary immediate EM; natural final EM and search count',
        dtype='bfloat16',do_sample=False,gpu_uuid=GPU,seed=2026091506,
        expected_trajectories=len(checkpoints)*(len(controlled)*len(CONDITIONS)+len(natural)),
        expected_forced=len(checkpoints)*len(controlled),
        source_hashes={str(p):checksum(p) for p in [Path(__file__),ROOT/'scripts/experiments/evidence_stopping/prepare.py',
            ROOT/'scripts/experiments/evidence_stopping/analyze.py',OUT/'EXPERIMENT_PLAN.md',Path(M.__file__)]},
        weights={})
    for c in checkpoints:
        p=ROOT/c['path']/'model.safetensors';s=p.stat();protocol['weights'][str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
    return controlled,natural,protocol


def run(smoke=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise RuntimeError('Only GPU 5 is authorized')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(2026091506);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==1
    controlled,natural,protocol=inputs_and_protocol()
    frozen=OUT/'protocol.frozen.json'
    if frozen.exists():assert M.read(frozen)==json.loads(json.dumps(protocol))
    else:M.save(frozen,protocol)
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='left'
    retriever=Retriever();retriever.search('Albert Einstein')
    if smoke:
        # Dedicated fabricated smoke item; never a scientific observation.
        model=AutoModelForCausalLM.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
        synthetic={'prompt':[dict(role='user',content='Answer within <answer> and </answer>. What is two plus two?')]}
        ids=tok.apply_chat_template(synthetic['prompt'],tokenize=True,add_generation_prompt=True)
        a=generate(model,tok,[ids,ids]);assert a[0]['raw_text']==a[1]['raw_text']
        thought='<think>Two plus two equals four.</think>\n'
        fake=dict(raw_text=thought+'<search>arithmetic</search>',token_ids=tok.encode(thought+'<search>arithmetic</search>',add_special_tokens=False))
        p=force_prefix(tok,ids,fake);assert p and p['reasoning_text']==thought
        b=generate(model,tok,[p['ids']],forced=True);assert forced_answer(b[0]) is not None
        M.save(OUT/'GPU_SMOKE.json',dict(passed=True,synthetic=True,normal=a[0],forced=b[0]))
        print('GPU_SMOKE_PASS',flush=True);return
    trajectory_path=OUT/'trajectories.jsonl';forced_path=OUT/'forced_answers.jsonl'
    trajectories=rows(trajectory_path);forced=rows(forced_path)
    done={(r['label'],r['qid'],r['condition']) for r in trajectories}
    forced_done={(r['label'],r['qid']) for r in forced}
    first_store={(r['label'],r['qid'],r['condition']):r for r in rows(OUT/'first_actions.jsonl')}
    t0=time.time()
    def progress(label):
        summarize(trajectories,forced)
        status=dict(state='running',checkpoint=label,trajectories_completed=len(trajectories),
            trajectories_total=protocol['expected_trajectories'],forced_completed=len(forced),
            forced_total=protocol['expected_forced'],elapsed_seconds=time.time()-t0,
            updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        M.save(OUT/'status.json',status);print('PROGRESS',json.dumps(status),flush=True)
    for ckpt in protocol['checkpoints']:
        label=ckpt['label'];tasks=[dict(q=q,condition=c) for q in controlled for c in CONDITIONS]
        tasks += [dict(q=q,condition='natural') for q in natural]
        todo=[t for t in tasks if (label,t['q']['qid'],t['condition']) not in done]
        force_todo=[q for q in controlled if (label,q['qid']) not in forced_done]
        if not todo and not force_todo:continue
        print('LOAD',label,flush=True)
        model=AutoModelForCausalLM.from_pretrained(ROOT/ckpt['path'],local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
        for start in range(0,len(todo),BATCH):
            completed=trajectory_batch(model,tok,retriever,todo[start:start+BATCH],label,first_store)
            for r in completed:append(trajectory_path,r);trajectories.append(r)
            progress(label)
        for start in range(0,len(force_todo),BATCH):
            cases=force_todo[start:start+BATCH];jobs=[];records=[]
            for q in cases:
                original=first_store[label,q['qid'],'both_hops'];p=force_prefix(tok,original['input_ids'],original['generation'])
                r=dict(label=label,qid=q['qid'],available=p is not None,
                    ordinary_action=original['action'],ordinary_content=original['content'],
                    ordinary_immediate_em=M.exact(original['content'] if original['action']=='answer' else None,q['gold']),
                    forced_answer=None,forced_em=False,forced_alias_em=False)
                records.append(r)
                if p is not None:jobs.append((q,r,p))
            if jobs:
                gs=generate(model,tok,[p['ids'] for _,_,p in jobs],forced=True)
                for (q,r,p),g in zip(jobs,gs):
                    answer=forced_answer(g);r.update(forced_answer=answer,forced_em=M.exact(answer,q['gold']),
                        forced_alias_em=M.exact(answer,q['aliases']),generation=g,**p)
            for r in records:append(forced_path,r);forced.append(r)
            progress(label)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(trajectories)==protocol['expected_trajectories'] and len(forced)==protocol['expected_forced']
    M.save(OUT/'status.json',dict(state='complete',trajectories_completed=len(trajectories),forced_completed=len(forced),elapsed_seconds=time.time()-t0))
    print('COMPLETE',len(trajectories),len(forced),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:run(args.smoke)
    except Exception as e:
        M.save(OUT/('smoke_failure.json' if args.smoke else 'failure.json'),dict(error=repr(e),at=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
