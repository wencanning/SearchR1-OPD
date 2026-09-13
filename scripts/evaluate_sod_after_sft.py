"""Wait for the identified SFT evaluation, then evaluate SOD on GPU 0 once."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT / os.environ.get('SOD_EVAL_OUT', 'eval_loop_logs/sod_15b_after_sft_20260912')
OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
MODEL = ROOT / 'verl_checkpoints/sod-grpo-coef0.5-1B/actor/global_step_150'
SFT_LOG = ROOT / os.environ.get('SFT_EVAL_LOG', 'eval_loop_logs/sft_15b_retry_20260912/eval.log')
TASKS = {'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'}

def metrics(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text(errors='replace'))
    text = re.sub(r'\(main_task pid=\d+\)\s*', '', text)
    text = text.replace("'", '').replace('"', '')
    pairs = re.findall(r'val/test_score/([\w/-]+)\s*:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)', text)
    result = dict((k, float(v)) for k, v in pairs)
    if not TASKS.issubset(result):
        raise RuntimeError(f'Incomplete evaluation results in {path}: {sorted(result)}')
    return result

def main():
    if (OUT / 'completed.json').exists():
        print('SOD evaluation already completed; refusing duplicate launch.', flush=True)
        return
    from safetensors import safe_open
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        with safe_open(str(MODEL / name), framework='pt') as weights:
            assert list(weights.keys())
    target = Path('/proc') / os.environ.get('SFT_EVAL_PID', '188331')
    identity = (target / 'stat').read_text().split(') ', 1)[1].split()[19] if target.exists() else None
    print(f'Waiting for SFT evaluation PID {target.name}; log: {SFT_LOG}; next model: {MODEL}', flush=True)
    while target.exists():
        try:
            if (target / 'stat').read_text().split(') ', 1)[1].split()[19] != identity:
                break
        except FileNotFoundError:
            break
        time.sleep(60)
    sft_metrics = metrics(SFT_LOG)
    (OUT / 'sft_prerequisite.json').write_text(json.dumps(sft_metrics, indent=2))
    print('SFT final metrics verified. Waiting for GPU 0 memory release.', flush=True)
    while True:
        used = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if used < 1000:
            break
        time.sleep(60)
    request = urllib.request.Request('http://127.0.0.1:8000/retrieve', data=json.dumps({'queries': ['HotpotQA'], 'topk': 1, 'return_scores': True}).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=60) as response:
        assert json.load(response)['result']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', N_GPUS='1', EVAL_SEED='42', EVAL_DO_SAMPLE='true', BASE_MODEL=str(MODEL), PYTHON_BIN=sys.executable, PYTHONPATH=str(ROOT), NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    print('Starting SOD 1.5B step 150 evaluation on GPU 0.', flush=True)
    with (OUT / 'eval.log').open('w') as log:
        subprocess.run(['bash', 'scripts/nq_hotpotqa/evaluate.sh',
                        'actor_rollout_ref.rollout.log_prob_micro_batch_size=8',
                        'actor_rollout_ref.ref.log_prob_micro_batch_size=8'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {'model': str(MODEL), 'metrics': metrics(OUT / 'eval.log')}
    (OUT / 'completed.json').write_text(json.dumps(result, indent=2))
    print('SOD evaluation completed:', json.dumps(result), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (OUT / 'failed.json').write_text(json.dumps({'error': repr(error)}, indent=2))
        raise
