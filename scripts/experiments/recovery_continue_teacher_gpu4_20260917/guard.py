"""Run one owned worker on physical GPU4; preserve shared retriever and others."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
GPU='GPU-209eafd7-38f5-a04a-12d5-8d316ea4f8a8'


def free():
    row=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    idx,mem=map(int,row.strip().split(','));assert idx==4
    return mem


def main():
    script=Path(sys.argv[1]).resolve();args=sys.argv[2:]
    assert script.parent==Path(__file__).resolve().parent
    if free()<34*1024:raise RuntimeError('GPU4 startup free memory below34GiB; not launched')
    out=Path(args[args.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,RECOVERY_GPU_GUARDED='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    p=subprocess.Popen([sys.executable,str(script),*args],env=env,start_new_session=True)
    try:
        with (out/'gpu_watchdog.jsonl').open('a') as f:
            while p.poll() is None:
                mem=free();lines=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
                own=sum(int(line.split(',')[1]) for line in lines if line.split(',')[0].strip()==str(p.pid))
                f.write(json.dumps(dict(time=time.time(),pid=p.pid,own_mib=own,card_free_mib=mem))+'\n');f.flush()
                if mem<16*1024 or own>=22*1024:raise RuntimeError('GPU4 memory guard; stop ONLY our worker')
                time.sleep(1)
    except BaseException:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
    raise SystemExit(p.returncode)


if __name__=='__main__':main()
