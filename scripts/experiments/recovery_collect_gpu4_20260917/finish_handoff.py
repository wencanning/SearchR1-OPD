"""Import completed GPU4 first states and resume the original GPU3 waiter.

On collection failure, resume original GPU3 collection instead. Never launch
GPU3 directly: the unchanged waiter still requires luorongchuan's job to exit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
QUEUE=ROOT/'reports/recovery_queue_20260917'
SOURCE=ROOT/'reports/recovery_collect_gpu4_20260917/collected'
TOKEN='gpu4-collection-handoff-20260917\n'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collect-exit',type=int,required=True);args=ap.parse_args()
    # Do not remove a STOP subsequently changed by the user.
    stop=QUEUE/'STOP'
    if not stop.exists() or stop.read_text()!=TOKEN:
        raise RuntimeError('Handoff pause token changed; preserve user control and do not auto-resume')
    # Original waiter must have stopped before result import/relaunch.
    old=json.loads((QUEUE/'queue_launch.json').read_text())['pid']
    for _ in range(8):
        try:
            cmd=Path(f'/proc/{old}/cmdline').read_bytes()
            alive=b'recovery_queue_20260917/wait_and_run.py' in cmd
        except FileNotFoundError: alive=False
        if not alive: break
        time.sleep(10)
    else: raise RuntimeError('Original waiter has not stopped; do not import/duplicate')
    record=dict(phase='resuming_original_gpu3_queue',gpu4_collect_exit=args.collect_exit,imported=False)
    if args.collect_exit==0:
        status=json.loads((SOURCE/'status.json').read_text())
        inp=json.loads((QUEUE/'train_inputs.frozen.json').read_text())
        rows=[json.loads(l) for l in (SOURCE/'states.jsonl').read_text().splitlines()]
        assert status['state']=='complete' and status['completed']==status['total']==len(inp['cases'])==128
        assert len(rows)==128 and {r['qid'] for r in rows}=={c['qid'] for c in inp['cases']}
        assert all(r['split']=='train' and r['model']=='student' for r in rows)
        mf=json.loads((SOURCE/'manifest.frozen.json').read_text())
        assert mf['source_sha256']==sha(QUEUE/'train_inputs.frozen.json')
        assert mf['gpu']=='GPU-209eafd7-38f5-a04a-12d5-8d316ea4f8a8'
        dest=QUEUE/'collected';dest.mkdir(exist_ok=True)
        copies={}
        for name in ['states.jsonl','manifest.frozen.json','retrieval_cache.json']:
            p=SOURCE/name
            if not p.exists(): continue
            target=dest/name
            if target.exists(): assert sha(target)==sha(p),f'Refuse conflicting {target}'
            else:
                tmp=target.with_suffix('.handoff.tmp');tmp.write_bytes(p.read_bytes());tmp.replace(target)
            copies[name]=sha(p)
        save(dest/'GPU4_REUSE_PROVENANCE.json',dict(source=str(SOURCE),sha256=copies,
            scope='Actual first-state collection on GPU4 copied byte-for-byte; not a second independent collection. GPU3 will skip completed collection and start continuation after its own wait gate.'))
        save(dest/'status.json',dict(status,collected_on_gpu4=True))
        record.update(imported=True,eligible=sum(r['eligible'] for r in rows))
    # Failed collection is never partially imported; original GPU3 stage starts fresh.
    stop.unlink()
    argv=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/experiments/recovery_queue_20260917/wait_and_run.py'),
          '--manifest',str(QUEUE/'queue_manifest.frozen.json')]
    with (QUEUE/'queue.log').open('a') as log:
        p=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(queue_pid=p.pid,resumed_at=time.strftime('%Y-%m-%d %H:%M:%S%z'))
    save(QUEUE/'queue_resume_after_gpu4.json',record)
    save(QUEUE/'gpu4_collection_handoff.json',record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__': main()
