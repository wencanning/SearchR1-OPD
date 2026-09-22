"""Called by the student handoff BEFORE allowing the original GPU3 queue to resume.

GPU4 teacher completion is imported once. A failed/dead teacher is not imported;
the unchanged original GPU3 waiter will run that stage after luorongchuan exits.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]
QUEUE=ROOT/'reports/recovery_queue_20260917'
OUT=ROOT/'reports/recovery_continue_teacher_gpu4_20260917'
TOKEN='gpu4-continuation-handoff-20260917\n'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)


def same_process(expected):
    try:
        p=Path('/proc')/str(expected['pid'])
        fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return fields[0]!='Z' and fields[19]==expected['start_ticks']
    except (FileNotFoundError,ProcessLookupError):return False


def import_complete(source,queue):
    status=json.loads((source/'status.json').read_text())
    inp=json.loads((queue/'train_inputs.frozen.json').read_text())
    expected={(c['qid'],k) for c in inp['cases'] for k in range(inp['replicates'])}
    rows=[json.loads(l) for l in (source/'continuations.jsonl').read_text().splitlines()]
    if status['state']!='complete' or status['completed']!=status['total'] or status['total']!=len(expected):
        raise ValueError('Teacher collection incomplete')
    if len(rows)!=len(expected) or {(r['qid'],r['replicate']) for r in rows}!=expected:
        raise ValueError('Teacher row membership mismatch')
    if any(r['model']!='teacher' or r['split']!='train' for r in rows):raise ValueError('Teacher role/split mismatch')
    mf=json.loads((source/'manifest.frozen.json').read_text())
    if mf['source_sha256']!=sha(queue/'train_inputs.frozen.json') or mf['states_sha256']!=sha(queue/'collected/states.jsonl'):
        raise ValueError('Teacher input/state hash mismatch')
    if mf['gpu']!='GPU-209eafd7-38f5-a04a-12d5-8d316ea4f8a8':raise ValueError('Wrong teacher GPU provenance')
    dest=queue/'continue_teacher';dest.mkdir(exist_ok=True);copies={}
    for name in ['continuations.jsonl','manifest.frozen.json','retrieval_cache.json']:
        p=source/name
        if not p.exists():continue
        target=dest/name
        if target.exists():
            if sha(target)!=sha(p):raise ValueError(f'Refuse conflicting {target}')
        else:
            tmp=target.with_suffix('.handoff.tmp');tmp.write_bytes(p.read_bytes());tmp.replace(target)
        copies[name]=sha(p)
    save(dest/'GPU4_REUSE_PROVENANCE.json',dict(source=str(source),sha256=copies,
        scope='Same frozen train states, teacher continuations copied byte-for-byte from GPU4; not independent duplicate samples'))
    save(dest/'status.json',dict(status,teacher_continuations_on_gpu4=True))
    return dict(state='teacher_imported',records=len(rows))


def main():
    dep=QUEUE/'gpu4_teacher_dependency.json'
    if not dep.exists():return
    expected=json.loads(dep.read_text())
    while True:
        allowed_tokens={TOKEN}
        if (QUEUE/'DISABLE_GPU3').exists():allowed_tokens.add('USER_CANCELLED_GPU3\n')
        if not (QUEUE/'STOP').exists() or (QUEUE/'STOP').read_text() not in allowed_tokens:
            raise RuntimeError('User pause token changed; do not resume queue')
        path=OUT/'supervisor_status.json'
        status=json.loads(path.read_text()) if path.exists() else {}
        if status.get('state') in ('complete','failed'):
            result=import_complete(OUT/'continue_teacher',QUEUE) if status['state']=='complete' else dict(state='teacher_failed_gpu3_fallback',supervisor=status)
            break
        if not same_process(expected):
            result=dict(state='teacher_supervisor_gone_gpu3_fallback')
            # A dead supervisor with a live child must NOT allow duplicate work.
            lockpath=OUT/'continue_teacher/run.lock'
            if lockpath.exists():
                import fcntl
                with lockpath.open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        save(QUEUE/'gpu4_teacher_handoff.json',dict(state='waiting_for_gpu4_teacher',pid=expected['pid']))
        time.sleep(20)
    if (QUEUE/'DISABLE_GPU3').exists():
        result['gpu3_fallback_disabled']=True
        result['state']=result['state'].replace('_gpu3_fallback','_no_gpu3_fallback')
    save(QUEUE/'gpu4_teacher_handoff.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
