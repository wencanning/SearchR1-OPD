#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
source /data/home/wencanning/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
mkdir -p experiment_logs/sft
exec 9>experiment_logs/sft/er_opd_after_sft_20260911.lock
flock -n 9 || { echo 'This queued job is already running.'; exit 1; }
python -u - <<'PY' 2>&1 | tee -a experiment_logs/sft/er_opd_after_sft_20260911.log
import datetime
import hashlib
import json
import pathlib
import re
import time
import urllib.request
from safetensors import safe_open

def log(message):
    print(datetime.datetime.now().astimezone().isoformat(), message, flush=True)

process = pathlib.Path('/proc/662620/cmdline')
experiment = 'sft-05B-teacher7B-hotpotqa-8k-gpu23-20260911'
log('WAITING for SFT ' + experiment + '; next job: bash train_er_opd.sh')
while process.exists():
    try:
        command = process.read_bytes()
    except FileNotFoundError:
        break
    if experiment.encode() not in command:
        raise RuntimeError('SFT PID was reused; refusing to infer completion')
    log('SFT still running; check again in 60 seconds')
    time.sleep(60)

checkpoint = pathlib.Path('verl_checkpoints') / experiment / 'global_step_375'
training_log = pathlib.Path('experiment_logs/sft/gpu23_20260911_tmux.log').read_text()
assert re.search(r'step:375\s*-\s*val/loss:\s*[0-9]', training_log), 'Missing final SFT validation'
assert 'Traceback (most recent call last)' not in training_log, 'SFT traceback found'
assert (checkpoint / 'config.json').is_file(), 'Final SFT checkpoint missing'
assert (checkpoint / 'tokenizer_config.json').is_file(), 'Final tokenizer missing'
weights = list(checkpoint.glob('*.safetensors'))
assert weights, 'Final SFT weights missing'
for path in weights:
    with safe_open(str(path), framework='pt', device='cpu') as f:
        assert list(f.keys()), 'Empty checkpoint'
assert hashlib.sha256(pathlib.Path('train_er_opd.sh').read_bytes()).hexdigest() == 'e80cf0e2a4d9f9bea4b61a5dc006aff24738fd8335b00079067b4dd484e006c6', 'ER script changed since queue setup; review before launching'
request = urllib.request.Request('http://127.0.0.1:8000/retrieve',
    data=json.dumps({'queries': ['HotpotQA'], 'topk': 1, 'return_scores': True}).encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=60) as response:
    assert json.load(response)['result'], 'Retriever returned no result'
log('READY: SFT final checkpoint and validation verified; retriever healthy; starting ER-OPD alpha=0.5 lambda=0.01')
PY
bash train_er_opd.sh 2>&1 | tee -a experiment_logs/er_opd_after_sft_20260911.log
