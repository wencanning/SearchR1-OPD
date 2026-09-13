#!/usr/bin/env bash
# Wait for the current 1.5B SFT, then evaluate final 0.5B and 1.5B checkpoints once each.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN=${PYTHON_BIN:-/data/home/wencanning/miniconda3/envs/searchr1/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export N_GPUS=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export EVAL_SEED=${EVAL_SEED:-42} EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE:-true}
export RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
export SFT_RUN_15B=sft-15B-teacher7B-hotpotqa-8k-gpu0-20260911
export MODEL_05B=verl_checkpoints/sft-05B-teacher7B-hotpotqa-8k-gpu23-20260911/global_step_375
export MODEL_15B=verl_checkpoints/$SFT_RUN_15B/global_step_375
RESULT_DIR=${RESULT_DIR:-eval_loop_logs/sft_comparison_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RESULT_DIR"
exec 9>experiment_logs/sft/evaluate_after_sft.lock
flock -n 9 || { echo 'An SFT evaluation queue is already active.'; exit 1; }
"$PYTHON_BIN" -u - <<'PY' 2>&1 | tee "$RESULT_DIR/wait.log"
import datetime, json, os, pathlib, re, time, urllib.request
from safetensors import safe_open
target = os.environ['SFT_RUN_15B'].encode()
while True:
    active = []
    for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if b'verl.trainer.fsdp_sft_trainer' in command and target in command:
            active.append(path.parent.name)
    if not active:
        break
    print(datetime.datetime.now().isoformat(), 'Waiting for 1.5B SFT:', active, flush=True)
    time.sleep(60)
log = pathlib.Path('experiment_logs/sft/gpu0_15b_20260911.log').read_text()
assert re.search(r'step:375\s*-\s*val/loss:\s*[0-9]', log), '1.5B SFT did not reach final validation'
assert 'Traceback (most recent call last)' not in log, '1.5B SFT reported an error'
for name in ('MODEL_05B', 'MODEL_15B'):
    directory = pathlib.Path(os.environ[name])
    assert (directory / 'config.json').is_file(), f'Missing config: {directory}'
    assert (directory / 'tokenizer_config.json').is_file(), f'Missing tokenizer: {directory}'
    files = list(directory.glob('*.safetensors'))
    assert files, f'Missing weights: {directory}'
    for file in files:
        with safe_open(str(file), framework='pt', device='cpu') as f:
            assert list(f.keys()), f'Empty weights: {file}'
request = urllib.request.Request(os.environ['RETRIEVER_URL'],
    data=json.dumps({'queries':['HotpotQA'], 'topk':1, 'return_scores':True}).encode(),
    headers={'Content-Type':'application/json'})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=60) as response:
    assert json.load(response)['result']
print('Final checkpoints verified; retriever ready. Starting sequential evaluation.', flush=True)
PY
for model_name in SFT_0.5B SFT_1.5B; do
    if [[ "$model_name" == SFT_0.5B ]]; then
        export BASE_MODEL="$MODEL_05B"
    else
        export BASE_MODEL="$MODEL_15B"
    fi
    echo "Evaluating $model_name: $BASE_MODEL on GPU $CUDA_VISIBLE_DEVICES"
    bash scripts/nq_hotpotqa/evaluate.sh "$@" 2>&1 | tee "$RESULT_DIR/$model_name.log"
    "$PYTHON_BIN" - "$RESULT_DIR/$model_name.log" "$RESULT_DIR/$model_name.json" "$BASE_MODEL" <<'PY'
import json, pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
pairs = re.findall(r"['\"]?(val/test_score/[A-Za-z0-9_/-]+)['\"]?\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
assert pairs, 'Evaluation exited without val/test_score metrics'
pathlib.Path(sys.argv[2]).write_text(json.dumps({'model':sys.argv[3], 'metrics':{k:float(v) for k,v in pairs}}, indent=2))
PY
done
echo "Both evaluations completed: $RESULT_DIR"
