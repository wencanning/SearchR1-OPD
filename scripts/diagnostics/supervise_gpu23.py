"""User-authorized tmux0 GPU2/3 continuation queue; never displaces foreign jobs.

Adopts existing jobs without restarting them, then launches distinct pure-OPD
seeds until a STOP file. Process identity includes /proc start ticks. No ray stop,
pkill, file deletion, or signals to the shared retriever are used.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiment_logs/gpu23-autoloop-20260914'
RETRIEVER=110019
UUIDS={2:'GPU-79f08c8c-6fe7-4ab0-d16f-00ab0dba95df',3:'GPU-4491c0be-1ea7-eff5-5e5b-9e4fe635f053'}


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None


def cmdline(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')


def process_table():
    raw=subprocess.check_output(['ps','-eo','pid=,ppid=,pgid='],text=True)
    return {int(p):(int(parent),int(group)) for p,parent,group in (line.split() for line in raw.splitlines())}


def descendants(root,table):
    result={root}
    while True:
        more={p for p,(parent,_) in table.items() if parent in result}
        if more<=result:return result
        result|=more


def gpu_pids():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    return {gpu:{int(p.strip()) for u,p in (line.split(',') for line in raw.splitlines()) if u.strip()==uuid}
            for gpu,uuid in UUIDS.items()}


def foreground(shell):
    fields=Path(f'/proc/{shell}/stat').read_text().split(') ',1)[1].split()
    return int(fields[5])


def event(event_type,**values):
    row=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),event=event_type,**values)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def atomic_state(data):
    temp=OUT/'state.tmp'
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(OUT/'state.json')


def tail(path,size=196608):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-size));return f.read().decode(errors='replace')
    except FileNotFoundError:return ''


def cleanup_known(job):
    # Freeze the exact identities observed while the adopted/owned root was alive.
    targets={int(p):birth for p,birth in job['known'].items() if birth is not None and int(p)!=RETRIEVER}
    assert os.getpid() not in targets and job['shell'] not in targets
    live=[p for p,b in targets.items() if identity(p)==b]
    for p in live:
        try:os.kill(p,signal.SIGTERM)
        except ProcessLookupError:pass
    if live:event('cleanup_own_descendants',gpu=job['gpu'],pids=live)
    # Subsequent polls recheck identities; never broaden cleanup to a GPU/user name.


def refresh_research_queue():
    """Predeclared result-driven dispatch, not invented claims or unbounded fixes."""
    path=ROOT/'reports/search_loop_20260914/generations_v2.jsonl'
    queue_path=OUT/'priority_queue.json'
    queue=json.loads(queue_path.read_text()) if queue_path.exists() else []
    if any(t['id']=='action-credit-20260914' for t in queue) or not path.exists():return
    by_model={name:{} for name in ['teacher_fp16','opd75','opd100']}
    for line in path.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r['model'] in by_model and r['kind']=='repeat' and r['state']=='repeat_return' and r['seed'] is None:
            by_model[r['model']][r['key']]=r
    common=set.intersection(*(set(v) for v in by_model.values()))
    if len(common)<64:return  # Do not act on a convenient partial pilot.
    parity={}
    for line in (ROOT/'reports/search_loop_20260914/scores_v2.jsonl').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r['model']=='teacher_fp16' and r['state']=='repeat_return':parity[r['key']]=r.get('numerical_pass',False)
    usable={k for k in common if parity.get(k,False)}
    selected=usable if usable else common
    rates={m:sum(v[k]['repeat'] is True for k in selected)/len(selected) for m,v in by_model.items()}
    if len(usable)<48:
        branch='precision_or_prefix_check';reason='同精度一致性不足，优先检查原始token路径和精度，不作机制归因'
    elif rates['teacher_fp16']>=.25:
        branch='teacher_loop';reason='教师在固定状态也常重复，先检查教师动作路径与局部目标，不预设学生独有故障'
    elif rates['opd100']>=rates['teacher_fp16']+.20:
        branch='student_transfer_or_update';reason='后期学生比教师更常重复；检查协议分词与局部reverse-KL/采样梯度，不直接宣称根因'
    else:
        branch='nonreplication_or_sampling';reason='当前greedy对照未显示足够大的教师学生差异；检查token路径并等待采样结果'
    task=dict(id='action-credit-20260914',kind='action_credit',gpus=[2,3],priority=100,
              branch=branch,reason=reason,n_common=len(common),n_numerically_usable=len(usable),repeat_rates=rates)
    queue.append(task);temp=OUT/'priority_queue.tmp';temp.write_text(json.dumps(queue,ensure_ascii=False,indent=2));temp.replace(queue_path)
    event('research_followup_queued',**task)


def refresh_protocol_screen():
    source=ROOT/'reports/action-credit-recovery-20260914/priming_cpu.jsonl'
    if not source.exists():return
    observations={}
    for line in source.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        observations[r['key']]=r
    if len(observations)<8:return
    flips=sum(r['same_text_alt_path_post']['canonical_first_logp']<r['original_post']['canonical_first_logp']-5
              for r in observations.values())
    if flips<6:return
    path=OUT/'priority_queue.json';queue=json.loads(path.read_text()) if path.exists() else []
    for mask,arm in [('false','baseline'),('true','masked')]:
        run_id=f'protocol-screen-{arm}-from75-20260914'
        if any(t['id']==run_id for t in queue):continue
        task=dict(id=run_id,kind='protocol_ablation',gpus=[2,3],priority=200,mask=mask,
                  reason=f'同文本token路径干预{flips}/8翻转教师标签支持；从同一OPD75权重对照协议监督开/关',
                  evidence=str(source),updates=40,seed=42)
        queue.append(task);event('protocol_causal_screen_queued',**task)
    temp=OUT/'priority_queue.tmp';temp.write_text(json.dumps(queue,ensure_ascii=False,indent=2));temp.replace(path)


def launch(job,jobs):
    gpu=job['gpu'];pane=job['pane'];shell=job['shell']
    assert foreground(shell)==os.getpgid(shell), 'Refuse to type into a running pane'
    assert gpu_pids()[gpu]<={RETRIEVER}, 'Foreign GPU process; no launch'
    if shutil.disk_usage(ROOT).free<100*2**30:raise RuntimeError('Less than 100GiB disk; no deletion and no new run')
    queue_path=OUT/'priority_queue.json'
    tasks=json.loads(queue_path.read_text()) if queue_path.exists() else []
    attempted={t for j in jobs.values() for t in j.get('attempted_tasks',[])}
    pending=sorted([t for t in tasks if t['id'] not in attempted and gpu in t['gpus']],key=lambda t:-t['priority'])
    task=pending[0] if pending else None
    seed=job['next_seed'];stamp=time.strftime('%Y%m%d-%H%M%S')
    if task:
        tag=task['id']
        if task['kind']=='action_credit':
            assert task['id'] in ['action-credit-20260914','action-credit-bf16-20260914']
            marker='run_action_credit.sh';command=f'bash scripts/diagnostics/run_action_credit.sh {gpu} {tag}'
            log=str(ROOT/f'reports/{tag}/run.log')
        elif task['kind']=='protocol_ablation':
            assert tag in ['protocol-screen-baseline-from75-20260914','protocol-screen-baseline-from75-retry1-20260914','protocol-screen-masked-from75-20260914']
            assert task['mask'] in ['true','false']
            marker='run_protocol_ablation.sh';command=f'bash scripts/diagnostics/run_protocol_ablation.sh {gpu} {tag} {task["mask"]}'
            log=str(ROOT/f'verl_checkpoints/pure-opd-05B-n1-{tag}/train.log')
        else:raise ValueError('Unknown research task kind')
    else:
        tag=f'autoloop-gpu{gpu}-seed{seed}-{stamp}';marker='train_pure_opd_sod.sh'
        command=(f'bash scripts/baseline/train_pure_opd_sod.sh opd {gpu} {tag} '
                 f'data.train_seed={seed} actor_rollout_ref.rollout.seed={seed} '
                 "trainer.logger='[console]' trainer.total_training_steps=151 trainer.save_freq=10")
        log=str(ROOT/f'verl_checkpoints/pure-opd-05B-n1-{tag}/train.log')
    subprocess.run(['tmux','send-keys','-t',pane,'C-u'],check=True)
    subprocess.run(['tmux','send-keys','-t',pane,'-l',command],check=True)
    subprocess.run(['tmux','send-keys','-t',pane,'Enter'],check=True)
    for _ in range(30):
        pid=foreground(shell)
        if pid!=os.getpgid(shell) and identity(pid) is not None:
            # Wait until the shell has exec'd/started the expected script, not an unrelated command.
            if marker in cmdline(pid) and tag in cmdline(pid):break
        time.sleep(.1)
    else:raise RuntimeError('Launch not identified; refuse blind retry')
    job.update(root=pid,birth=identity(pid),known={str(pid):identity(pid)},started=time.time(),
               next_seed=seed+(0 if task else 2),tag=tag,status='running',owned=True,log=log,
               traceback_reported=False,completion_file=str(ROOT/f'reports/{tag}/COMPLETE.json') if task else None)
    if task:job.setdefault('attempted_tasks',[]).append(task['id'])
    event('launched_research_priority' if task else 'launched_pure_opd_fallback',gpu=gpu,pane=pane,pid=pid,
          seed=None if task else seed,tag=tag,command=command,reason=task['reason'] if task else '独立seed验证循环是否可复现；等待下一项机制实验')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu2-root',type=int,default=681560)
    parser.add_argument('--gpu3-root',type=int,default=2053522)
    parser.add_argument('--replace-gpu2-watcher',type=int,default=692278)
    parser.add_argument('--adopt-gpu3-recovery',type=int)
    parser.add_argument('--adopt-gpu3-run-id',default='action-credit-recovery-20260914',
                        choices=['action-credit-recovery-20260914','action-credit-bf16-20260914'])
    parser.add_argument('--resume-fixed-gpu3',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT)
    assert identity(RETRIEVER) is not None and 'retrieval' in cmdline(RETRIEVER), 'Shared retriever identity changed'
    if (OUT/'state.json').exists():
        previous=json.loads((OUT/'state.json').read_text())
        jobs={int(g):j for g,j in previous['jobs'].items()}
        event('resumed_controller',pid=os.getpid())
    else:
        jobs={}
        for gpu,pane,pid,marker,seed in [(2,'%1',args.gpu2_root,'train_pure_opd_sod.sh',100),
                                        (3,'%5',args.gpu3_root,'run_search_loop_queue.sh',101)]:
            assert identity(pid) is not None and marker in cmdline(pid)
            shell=int(subprocess.check_output(['tmux','display-message','-p','-t',pane,'#{pane_pid}']))
            assert foreground(shell)==pid and os.getpgid(pid)==pid and pid!=os.getpgid(shell)
            jobs[gpu]=dict(gpu=gpu,pane=pane,shell=shell,root=pid,birth=identity(pid),known={},
                           next_seed=seed,status='running',owned=False,started=time.time(),tag='adopted',
                           log=str(ROOT/('verl_checkpoints/pure-opd-05B-n1-20260914-1220-coef1/train.log' if gpu==2
                                         else 'reports/search_loop_20260914/gpu3.log')))
        watcher=args.replace_gpu2_watcher
        if identity(watcher) is not None:
            command=cmdline(watcher)
            assert 'watch_training_handoff.py' in command and '--pid 681560 --pane %1 --gpu 2' in command
            os.kill(watcher,signal.SIGTERM)
            event('replaced_er_grpo_watcher_only',pid=watcher)
        event('adopted_without_restart',pid=os.getpid(),gpu2=args.gpu2_root,gpu3=args.gpu3_root)
    if args.adopt_gpu3_recovery:
        pid=args.adopt_gpu3_recovery;job=jobs[3]
        recovery_tag=args.adopt_gpu3_run_id
        assert identity(job['root'])!=job['birth'], 'Old GPU3 job still alive'
        assert identity(pid) is not None and f'run_action_credit.sh 3 {recovery_tag}' in cmdline(pid)
        assert foreground(job['shell'])==pid and os.getpgid(pid)==pid
        job.update(root=pid,birth=identity(pid),known={},owned=True,started=time.time(),status='running',
            tag=recovery_tag,log=str(ROOT/f'reports/{recovery_tag}/run.log'),
            completion_file=str(ROOT/f'reports/{recovery_tag}/COMPLETE.json'),traceback_reported=False)
        job.setdefault('attempted_tasks',[]).append('action-credit-bf16-20260914' if 'bf16' in recovery_tag else 'action-credit-20260914')
        event('adopted_manual_research_recovery',gpu=3,pid=pid)
    if args.resume_fixed_gpu3:
        job=jobs[3]
        assert job['status'].startswith('blocked') and identity(job['root'])!=job['birth']
        assert gpu_pids()[3]<={RETRIEVER} and foreground(job['shell'])==os.getpgid(job['shell'])
        job['status']='handoff'
        event('resume_after_verified_startup_fix',gpu=3,fix='short unique Ray socket directory; preserve failed run')
    tick=0
    while True:
        try:
            if tick%12==0:
                refresh_research_queue()
                refresh_protocol_screen()
            table=process_table();gpu_map=gpu_pids()
            for gpu,job in jobs.items():
                stop=(OUT/'STOP').exists() or (OUT/f'STOP_GPU{gpu}').exists() or (ROOT/'reports/search_loop_20260914/STOP').exists()
                if identity(job['root'])==job['birth']:
                    for p in descendants(job['root'],table):
                        birth=identity(p)
                        if birth is not None:job['known'][str(p)]=birth
                    # A shell may have buffered its old fallback loop before the handoff file existed.
                    # If the diagnostic child has ended and research work is queued, do not spend
                    # three old fallback runs before returning the slot to the priority dispatcher.
                    if gpu==3 and job['tag']=='adopted':
                        metadata=ROOT/'reports/search_loop_20260914/run_metadata.json'
                        if 'diagnostic_child' not in job and metadata.exists():
                            child=int(json.loads(metadata.read_text())['pid'])
                            if str(child) in job['known']:
                                job['diagnostic_child']=dict(pid=child,birth=job['known'][str(child)])
                        child=job.get('diagnostic_child')
                        queue_path=OUT/'priority_queue.json'
                        pending=json.loads(queue_path.read_text()) if queue_path.exists() else []
                        attempted={t for j in jobs.values() for t in j.get('attempted_tasks',[])}
                        has_priority=any(t['id'] not in attempted and gpu in t['gpus'] for t in pending)
                        if child and identity(child['pid'])!=child['birth'] and has_priority and not stop:
                            assert 'run_search_loop_queue.sh' in cmdline(job['root'])
                            event('handoff_finished_diagnostic_before_legacy_fallback',gpu=gpu,root=job['root'])
                            cleanup_known(job)
                            continue
                    foreign=gpu_map[gpu]-{RETRIEVER}-{int(p) for p,b in job['known'].items() if identity(int(p))==b}
                    if foreign and job.get('foreign')!=sorted(foreign):event('foreign_process_no_displacement',gpu=gpu,pids=sorted(foreign))
                    job['foreign']=sorted(foreign);job['status']='running_stop_after_current' if stop else 'running'
                    text=tail(job['log']);steps=re.findall(r'(?:epoch \d+, step |step:)(\d+)',text)
                    if steps:job['latest_step']=int(steps[-1])
                    job['log_tail_has_traceback']='Traceback (most recent call last)' in text
                    if job['log_tail_has_traceback'] and not job.get('traceback_reported'):
                        event('traceback_in_live_log',gpu=gpu,log=job['log']);job['traceback_reported']=True
                    continue
                if job['status'].startswith('running'):
                    event('job_exited',gpu=gpu,root=job['root'],tag=job['tag'],log=job['log'])
                    cleanup_known(job);job['exited_at']=time.time();job['status']='handoff'
                    completed=bool(job.get('completion_file') and Path(job['completion_file']).exists())
                    if job['owned'] and time.time()-job['started']<300 and not completed:
                        job['status']='blocked_startup_failure';event('blocked_no_restart_storm',gpu=gpu,log=job['log'])
                if stop:
                    job['status']='stopped_by_file';continue
                if job['status'].startswith('blocked'):continue
                remaining=[int(p) for p,b in job['known'].items() if identity(int(p))==b]
                if remaining:
                    if time.time()-job.get('exited_at',time.time())>15:
                        for p in remaining:
                            assert p!=RETRIEVER and p!=os.getpid() and p!=job['shell']
                            try:os.kill(p,signal.SIGKILL)
                            except ProcessLookupError:pass
                        event('cleanup_stuck_own_descendants',gpu=gpu,pids=remaining)
                    continue
                foreign=gpu_pids()[gpu]-{RETRIEVER}
                if foreign:
                    if job.get('foreign')!=sorted(foreign):event('handoff_wait_foreign',gpu=gpu,pids=sorted(foreign))
                    job['foreign']=sorted(foreign);continue
                try:launch(job,jobs)
                except Exception as exc:
                    job['status']='blocked_launch';event('launch_blocked',gpu=gpu,error=repr(exc))
            atomic_state(dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),controller_pid=os.getpid(),jobs=jobs))
            if tick%60==0:
                # Store an honest partial summary; this is reporting, not automatic method invention.
                result=subprocess.run(['/data/home/wencanning/miniconda3/envs/searchr1/bin/python',
                    str(ROOT/'scripts/diagnostics/summarize_search_loop.py')],capture_output=True,text=True,timeout=30)
                (OUT/'latest_diagnostic_summary.txt').write_text(result.stdout+result.stderr)
            tick+=1
        except Exception as exc:
            event('controller_check_failed_no_mutation',error=repr(exc))
        time.sleep(5)


if __name__=='__main__':main()
