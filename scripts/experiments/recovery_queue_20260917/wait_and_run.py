"""Wait for identified GPU3 jobs; run a frozen, sequential collection manifest.

No CUDA context while waiting, no termination of pre-existing processes.
STOP prevents subsequent launches; it does not interrupt a running stage.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time


def identity(pid):
    try:
        p = Path(f'/proc/{pid}')
        fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z': return None
        return dict(pid=pid, start_ticks=fields[19], user=pwd.getpwuid(p.stat().st_uid).pw_name,
                    cmdline=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip())
    except (FileNotFoundError, ProcessLookupError): return None


def same_process(current, expected):
    return current is not None and all(current[k] == expected[k] for k in ['pid', 'start_ticks', 'user'])


def snapshot(gpu):
    data = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10)
    idx, uuid, free = [x.strip() for x in data.strip().split(',')]
    assert int(idx) == 3 and uuid == gpu
    raw = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
    processes = []
    for line in raw.splitlines():
        pid, mem = [x.strip() for x in line.split(',')]
        processes.append(dict(pid=int(pid), memory_mib=int(mem), identity=identity(int(pid))))
    return dict(free_mib=int(free), processes=processes)


def blockers(manifest, snap):
    reasons = []
    # The watched process must EXIT, even if it temporarily releases its GPU.
    for watched in manifest['wait_for']:
        if same_process(identity(watched['pid']), watched): reasons.append(f"waiting_pid:{watched['pid']}")
    permitted = manifest['shared_retriever']
    if not same_process(identity(permitted['pid']), permitted): reasons.append('shared_retriever_identity_changed_or_missing')
    for p in snap['processes']:
        if not same_process(p['identity'], permitted): reasons.append(f"gpu3_busy_pid:{p['pid']}")
    if snap['free_mib'] < manifest['minimum_start_free_mib']: reasons.append('insufficient_free_memory')
    return reasons


def save(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def validate_files(manifest):
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected: raise RuntimeError(f'Frozen input/code changed: {name}')


def complete(job):
    p = Path(job['expected_status'])
    if not p.exists(): return False
    obj = json.loads(p.read_text())
    return (obj.get('state') == 'complete' and isinstance(obj.get('completed'), int)
            and isinstance(obj.get('total'), int) and obj['completed'] == obj['total'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--check-only', action='store_true'); args = ap.parse_args()
    m = json.loads(args.manifest.read_text()); out = Path(m['output_dir'])
    validate_files(m)
    if args.check_only:
        snap = snapshot(m['gpu_uuid']); print(json.dumps(dict(snapshot=snap, blockers=blockers(m,snap)), indent=2)); return
    lock = (out / 'queue.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = Path(m['gpu_lock']).open('w'); fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(queue_pid=os.getpid(), state='waiting', completed_jobs=[], stage=None,
                 manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    statusfile = out / 'queue_state.json'
    def emit(**kw):
        state.update(kw, updated_at=time.strftime('%Y-%m-%d %H:%M:%S%z'))
        save(statusfile, state)
    try:
        for job in m['jobs']:
            if complete(job):
                state['completed_jobs'].append(job['name']); continue
            stable = 0; last_reason = None
            while True:
                if (out/'STOP').exists(): emit(state='stopped_before_launch', stage=job['name']); return
                try:
                    snap = snapshot(m['gpu_uuid']); reasons = blockers(m, snap)
                except (OSError, ValueError, subprocess.SubprocessError, AssertionError) as exc:
                    snap = {}; reasons = [f'probe_error:{type(exc).__name__}:{exc}']
                stable = stable+1 if not reasons else 0
                emit(state='waiting', stage=job['name'], blockers=reasons, stable_checks=stable, gpu=snap)
                if reasons != last_reason:
                    print(time.strftime('%H:%M:%S'), job['name'], 'WAIT', reasons or ['settling'], flush=True); last_reason=reasons
                if stable >= m['stable_checks']: break
                time.sleep(m['poll_seconds'])
            validate_files(m)
            # Recheck immediately before spawning, after hash validation.
            if blockers(m, snapshot(m['gpu_uuid'])):
                raise RuntimeError('GPU availability changed immediately before launch; no job started')
            emit(state='running', stage=job['name'], blockers=[])
            print(time.strftime('%H:%M:%S'), 'LAUNCH', job['name'], flush=True)
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            with (out/(job['name']+'.log')).open('a') as log:
                child = subprocess.Popen(job['argv'], cwd=m['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT)
                emit(worker_pid=child.pid)
                while child.poll() is None:
                    emit(state='running', stage=job['name']); time.sleep(10)
                if child.returncode != 0 or not complete(job):
                    emit(state='failed', returncode=child.returncode, reason='worker failed or expected complete status absent')
                    return
            state['completed_jobs'].append(job['name']); emit(worker_pid=None)
        emit(state='complete', stage=None, worker_pid=None)
        print('QUEUE_COMPLETE', flush=True)
    except BaseException as exc:
        emit(state='queue_error', reason=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__': main()
