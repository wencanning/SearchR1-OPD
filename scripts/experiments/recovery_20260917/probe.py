"""Fixed-answer continuation under evidence/history interventions; no training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
ROOT=Path(__file__).resolve().parents[3]
GPU='GPU-4491c0be-1ea7-eff5-5e5b-9e4fe635f053'
STUDENT=ROOT/'verl_checkpoints/opd-grpo-0.5B/actor/global_step_50'
TEACHER=Path('/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3')


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['student','teacher'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--witness',action='store_true');args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and os.environ.get('RECOVERY_GPU_GUARDED')=='1'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteria,LogitsProcessor
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
    path=STUDENT if args.model=='student' else TEACHER
    source=ROOT/'reports/recovery_20260917/probe_inputs.frozen.json';inp=json.loads(source.read_text());cases=inp['cases'][:1] if args.witness else inp['cases']
    args.out.mkdir(exist_ok=True,parents=True)
    import fcntl
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    weights={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in path.glob('*.safetensors')}
    manifest=dict(inputs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model=args.model,checkpoint=str(path),weights=weights,witness=args.witness,dtype='bfloat16',attention='sdpa',gpu=GPU)
    mf=args.out/'manifest.frozen.json'
    if mf.exists():assert json.loads(mf.read_text())==manifest
    else:save(mf,manifest)
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    check=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    assert tok.get_vocab()==check.get_vocab(),'Cannot reuse token prefix across different tokenizers'
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
    eos=list(set([tok.eos_token_id,tok.convert_tokens_to_ids('<|endoftext|>')]))
    class Vocab(LogitsProcessor):
        def __call__(self,input_ids,scores):scores[:,len(tok):]=-float('inf');return scores
    class Stop(StoppingCriteria):
        def __init__(self,n):self.n=n
        def __call__(self,ids,scores,**kwargs):return '</answer>' in tok.decode(ids[0,self.n:],skip_special_tokens=False)
    spec=importlib.util.spec_from_file_location('score',ROOT/'scripts/experiments/evidence_use/experiment.py');score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    result=args.out/'answers.jsonl';old=[json.loads(l) for l in result.read_text().splitlines()] if result.exists() else [];done={(r['qid'],r['variant'],r['replicate']) for r in old};assert len(done)==len(old)
    variants=['actual_native'] if args.witness else ['actual_native','actual_neutral','support_native','support_neutral'];reps=1 if args.witness else inp['replicates'];total=len(cases)*len(variants)*reps
    for case in cases:
        for rep in range(reps):
            order=variants if rep%2==0 else list(reversed(variants))
            for variant in order:
                key=(case['qid'],variant,rep)
                if key in done:continue
                seed=int(hashlib.sha256(f'20260917|{case["qid"]}|{rep}'.encode()).hexdigest()[:8],16)
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                ids=case['variants'][variant]['prefix_ids']+tok.encode('<answer>',add_special_tokens=False)
                x=torch.tensor([ids],device='cuda');start=time.monotonic()
                with torch.inference_mode():
                    gen=model.generate(x,attention_mask=torch.ones_like(x),do_sample=True,temperature=1.,top_p=1.,top_k=0,max_new_tokens=16 if args.witness else inp['max_new_tokens'],pad_token_id=tok.pad_token_id,eos_token_id=eos,logits_processor=[Vocab()],stopping_criteria=[Stop(len(ids))],use_cache=True,num_logits_to_keep=1)[0,len(ids):].tolist()
                end=next((i for i,t in enumerate(gen) if t in eos),len(gen));text=tok.decode(gen[:end],skip_special_tokens=False)
                match=re.fullmatch(r'\s*(.*?)</answer>\s*',text,re.S);valid=bool(match and not re.search(r'</?(?:think|search|answer|information)>',match[1]))
                answer=match[1].strip() if valid else None
                row=dict(qid=case['qid'],variant=variant,replicate=rep,seed=seed,model=args.model,answer=answer,em=score.exact(answer,case['gold']),valid=valid,raw_text=text,generated_ids=gen,prefix_tokens=len(ids),seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved())
                with result.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                done.add(key);save(args.out/'status.json',dict(state='running',completed=len(done),total=total));print('ANSWER',args.model,key,row['em'],row['valid'],flush=True)
    for file,meta in weights.items():
        stat=Path(file).stat();assert meta==dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    save(args.out/'status.json',dict(state='complete',completed=len(done),total=total,peak_cuda_allocated=torch.cuda.max_memory_allocated()))
    if args.witness:print('GPU3_RECOVERY_WITNESS_PASS',args.model,flush=True)


if __name__=='__main__':main()
