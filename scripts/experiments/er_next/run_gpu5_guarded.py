"""Manual, single-child GPU5 inference launcher; never stops another process."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GPU = 'GPU-2abc6681-2116-879c-a361-749edb32ac3b'


def memory():
    text = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10)
    index, free = [int(v.strip()) for v in text.strip().split(',')]
    if index != 5:
        raise RuntimeError('GPU UUID is no longer physical GPU5')
    return free


def main():
    args = sys.argv[1:]
    if not args or '--help' in args:
        print('Usage: python run_gpu5_guarded.py <collect_forks.py arguments excluding --device>')
        return
    if any(a == '--device' or a.startswith('--device=') for a in args):
        raise SystemExit('Device is fixed by this launcher.')
    if memory() < 20 * 1024:
        raise SystemExit('GPU5 has less than 20 GiB free; no job launched.')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, ER_GPU_GUARDED='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    cmd = [sys.executable, str(Path(__file__).with_name('collect_forks.py')), *args, '--device', 'cuda']
    child = subprocess.Popen(cmd, env=env, start_new_session=True)
    telemetry = None
    if '--out' in args:
        out = Path(args[args.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
        telemetry = (out / 'gpu_watchdog.jsonl').open('a')
    try:
        while child.poll() is None:
            free = memory()
            if free < 12 * 1024:
                raise RuntimeError('Free memory guard triggered; stopping only our worker.')
            records = subprocess.check_output(['nvidia-smi', '-i', GPU,
                '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
            own = 0
            for line in records.splitlines():
                parts = [v.strip() for v in line.split(',')]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == child.pid:
                    own += int(parts[1])
            if telemetry:
                telemetry.write(json.dumps(dict(time=time.time(),pid=child.pid,own_mib=own,card_free_mib=free))+'\n')
                telemetry.flush()
            if own >= 8 * 1024:
                raise RuntimeError('Own process reached 8 GiB; stopping only our worker.')
            time.sleep(1)
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
        raise
    finally:
        if telemetry:
            telemetry.close()
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
