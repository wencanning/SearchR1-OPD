"""Run one pinned teacher collection; record terminal status even on failure."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/recovery_continue_teacher_gpu4_20260917'


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=OUT/'launch_manifest.json')
    args=parser.parse_args()
    result=dict(state='failed',returncode=None,started_at=time.strftime('%Y-%m-%d %H:%M:%S%z'))
    try:
        manifest=json.loads(args.manifest.read_text())
        for name,expected in manifest['sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Frozen launch input changed: {name}')
        save(OUT/'supervisor_status.json',dict(result,state='running',pid=os.getpid()))
        p=subprocess.run(manifest['argv'],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        result['returncode']=p.returncode
        if p.returncode==0:
            status=json.loads((OUT/'continue_teacher/status.json').read_text())
            if status['state']!='complete' or status['completed']!=status['total'] or status['total']!=512:
                raise RuntimeError('Worker exited without complete512 records')
            result['state']='complete'
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        result['finished_at']=time.strftime('%Y-%m-%d %H:%M:%S%z')
        save(OUT/'supervisor_status.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['state']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
