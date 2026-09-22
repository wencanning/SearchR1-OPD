"""Explicit GPU 2/3 dispatcher. Default command is read-only preflight, never auto-launch."""
import argparse
import fcntl
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import DEFAULT_CONFIG, METHODS, ROOT, checkpoint, config, frozen, prepare, read, rows, run_dir, save, train_command, training_source


def devices():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    found={}
    for line in raw.splitlines():
        index,uuid,memory=[x.strip() for x in line.split(',')]
        if index in ('2','3'): found[int(index)]=dict(uuid=uuid,memory_mb=int(memory))
    if len(found)!=2: raise RuntimeError('Physical GPUs 2 and 3 must both exist')
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
    for index,d in found.items():
        d['processes']=[line for line in raw.splitlines() if line.split(',')[0].strip()==d['uuid']]
    return found


def idle_or_fail():
    ds=devices()
    for index,d in ds.items(): print(f'GPU {index}: {d}',flush=True)
    if any(d['processes'] or d['memory_mb']>1024 for d in ds.values()):
        raise RuntimeError('GPU 2/3 occupied. No job was stopped. Run again after your jobs release both devices.')
    return ds


def environment(uuids, seed=42):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONPATH=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_ATTENTION_BACKEND='XFORMERS',
        PYTHONHASHSEED=str(seed),SEARCHR1_RAY_CPUS='16')
    env.pop('RAY_ADDRESS',None)
    return env


def run_pair(c,a,ds,stage,label,steps,method,seed,limit=0):
    folder=Path(c['output'])/label; folder.mkdir(parents=True,exist_ok=True)
    frozen(folder/f'{stage}.request.json',dict(stage=stage,label=label,steps=steps,method=method,seed=seed,limit=limit))
    processes=[]; handles=[]
    try:
        for rank,index in enumerate((2,3)):
            cmd=[c['probe_python'],str(Path(__file__).with_name('worker.py')),'--config',str(Path(a.config).resolve()),
                 '--stage',stage,'--label',label,'--method',method,'--seed',str(seed),'--rank',str(rank),
                 '--steps',*map(str,steps),'--limit',str(limit)]
            log=(folder/f'{stage}-gpu{index}.log').open('a'); handles.append(log)
            print(shlex.join(cmd),flush=True)
            processes.append(subprocess.Popen(cmd,cwd=ROOT,env=environment([ds[index]['uuid']],seed),stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'{stage} worker failed; inspect {folder}/{stage}-gpu*.log. Completed rows retained.')
            time.sleep(2)
        if any(p.returncode for p in processes): raise RuntimeError(f'{stage} failed; inspect worker logs in {folder}')
    finally:
        # Only direct children owned by this invocation; never touch foreign GPU/Ray jobs.
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        for h in handles: h.close()
    save(folder/f'{stage}.complete.json',dict(completed=True,steps=steps,method=method,seed=seed,limit=limit))


def review_cards(c,label):
    folder=Path(c['output'])/label
    allrows=rows(folder/'candidates-0.jsonl')+rows(folder/'candidates-1.jsonl')
    cards=[]; template=[]
    for i,r in enumerate(sorted(allrows,key=lambda r:r['key'])):
        # Scores and ER outputs are deliberately absent. Identifiers are opaque in cards.
        rid=f'R{i+1:05d}'
        cards.append(dict(review_id=rid,question=r['question'],reference_answers=r['gold'],
              visible_observations=r['observations'],student_answer=r['answer'],
              baseline_answers={k:v['answer'] for k,v in r.get('teacher_generations',{}).items()},
              technical_exclusion=r.get('technical_exclusion')))
        template.append(dict(review_id=rid,key=r['key'],reviewed=False,reviewer='',sufficient=None,
             evidence_reason='',correct_aliases=r['gold'],wrong_classes=[],notes=''))
    frozen(folder/'review_cards.json',cards)
    frozen(folder/'annotations.template.json',template)
    print(f'Blind-review inputs: {folder}/review_cards.json\nCopy annotations.template.json to annotations.json and review ALL rows before score.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='check',choices=['prepare','check','pilot','collect','score','smoke','train','evaluate','summarize'])
    p.add_argument('--config',default=str(DEFAULT_CONFIG)); p.add_argument('--method',choices=METHODS,default='opd')
    p.add_argument('--seeds',type=int,nargs='+'); p.add_argument('--label'); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); c=config(a.config); os.chdir(ROOT); seeds=a.seeds or c['seeds']
    if any(s not in c['seeds'] for s in seeds): raise ValueError('Seeds must be declared in the frozen protocol')
    if a.stage in ('pilot','collect') and a.seeds and len(a.seeds)!=1:
        raise ValueError('Use one seed per fixed diagnostic reference; training/evaluation accept multiple seeds')
    if a.dry_run:
        if a.stage in ('train','smoke'):
            for seed in seeds[:1] if a.stage=='smoke' else seeds:
                for m in METHODS: print(shlex.join(train_command(c,m,seed,a.stage=='smoke')))
        else: print(dict(stage=a.stage,physical_gpus=[2,3],config=c,label=a.label,seeds=seeds))
        return
    if a.stage=='check':
        for k in ('student','teacher','train_file','test_file','training_python','probe_python'):
            print(k,c[k],Path(c[k]).exists())
        idle_or_fail(); return
    prepare(c)
    if a.stage=='prepare': return
    if a.stage=='summarize':
        from summarize import summarize
        summarize(c,a.label or f'diagnostic-{a.method}-seed{seeds[0]}'); return
    label=a.label or ('pilot' if a.stage=='pilot' else f'diagnostic-{a.method}-seed{seeds[0]}')
    if a.stage=='score':
        from worker import validate_annotation
        anns=read(Path(c['output'])/label/'annotations.json')
        for item in anns: validate_annotation(item)
        frozen(Path(c['output'])/label/'annotations.frozen.json',anns)
    locks=[]
    try:
        for gpu in (2,3):
            f=open(f'/tmp/searchr1-evidence-gap-gpu{gpu}.lock','w'); locks.append(f)
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ds=idle_or_fail()
        if a.stage in ('train','smoke','pilot','collect','evaluate'):
            import requests
            r=requests.post(c['retriever'],json={'queries':['HotpotQA preflight'],'topk':1,'return_scores':True},timeout=120)
            r.raise_for_status()
            if 'result' not in r.json(): raise RuntimeError('Unexpected retriever response')
        if a.stage in ('train','smoke'):
            source=training_source(c)
            for seed in seeds[:1] if a.stage=='smoke' else seeds:
                for m in METHODS:
                    out=run_dir(c,m,seed) if a.stage=='train' else Path(c['output'])/'smoke'/m
                    if (out/'COMPLETE.json').exists(): print('SKIP complete',out); continue
                    if (out/'train.log').exists():
                        raise RuntimeError(f'Incomplete training at {out}. Refusing an implicit optimizer reset; inspect it and use a new protocol output for a restart.')
                    ds=idle_or_fail(); out.mkdir(parents=True,exist_ok=True)
                    cmd=train_command(c,m,seed,a.stage=='smoke'); save(out/'command.json',cmd)
                    env=environment([ds[i]['uuid'] for i in (2,3)],seed)
                    env['PYTHONPATH']=str(source)+os.pathsep+str(ROOT)
                    env['PATH']=str(Path(c['training_python']).parent)+os.pathsep+env['PATH']
                    env['RAY_TMPDIR']=tempfile.mkdtemp(prefix='evidence-gap-ray-')
                    print('START',m,seed,'log:',out/'train.log',flush=True); start=time.time()
                    with (out/'train.log').open('w') as log:
                        subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    expected=[1] if a.stage=='smoke' else c['steps'][1:]
                    if any(not (out/'actor'/f'global_step_{s}'/'config.json').exists() for s in expected):
                        raise RuntimeError(f'Training exited without all planned checkpoints: {out}')
                    save(out/'COMPLETE.json',dict(method=m,seed=seed,steps=expected,wall_seconds=time.time()-start))
        elif a.stage=='evaluate':
            for m in METHODS:
                for seed in seeds:
                    for step in c['steps']: checkpoint(c,m,seed,step)
                    ds=idle_or_fail()
                    run_pair(c,a,ds,'evaluate',f'evaluation-{m}-seed{seed}',c['steps'],m,seed)
        elif a.stage=='score':
            collected=read(Path(c['output'])/label/'collect.request.json')
            run_pair(c,a,ds,'score',label,collected['steps'],collected['method'],collected['seed'])
            from summarize import summarize
            summarize(c,label)
        else:
            method='legacy' if a.stage=='pilot' else a.method
            steps=c['pilot_steps'] if a.stage=='pilot' else c['steps']
            for s in steps: checkpoint(c,method,seeds[0],s)
            run_pair(c,a,ds,'collect',label,steps,method,seeds[0],c['pilot_questions'] if a.stage=='pilot' else 0)
            review_cards(c,label)
    finally:
        for f in locks: f.close()

if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,FileNotFoundError,BlockingIOError) as e:
        print(f'STOP: {e}',file=sys.stderr); sys.exit(1)
