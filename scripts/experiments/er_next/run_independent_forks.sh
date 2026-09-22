#!/usr/bin/env bash
# User full run or bounded diagnostic; requires our independently started CPU service.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-help}"
case "$MODE" in
  pilot) LIMIT=2; REPLICATES=1 ;;
  small32) LIMIT=4; REPLICATES=4 ;;
  full) LIMIT=0; REPLICATES=4 ;;
  *) echo 'Usage: bash scripts/experiments/er_next/run_independent_forks.sh pilot|small32|full'; exit 2 ;;
esac
RUN_DIR="reports/er_next_20260916/forks_independent_${MODE}"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python - "$RUN_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
import requests
s=requests.Session();s.trust_env=False
r=s.get('http://127.0.0.1:18085/health',timeout=5);r.raise_for_status(); health=r.json()
assert health['cuda_initialized'] is False and health['encoder_dtype']=='torch.float32'
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
code=Path('scripts/experiments/er_next/cpu_retriever.py')
payload=dict(health=health,code_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
    env_spec_sha256=hashlib.sha256(Path('.aris/compute/fork-cpu-retriever-spec.json').read_bytes()).hexdigest())
p=out/'retriever.frozen.json'
if p.exists():
    previous=json.loads(p.read_text())
    # PID/loading time may change across legitimate restarts; scientific backend may not.
    for k in ('pid','seconds'): previous['health'].pop(k,None);payload['health'].pop(k,None)
    assert previous==payload, 'Independent retrieval backend changed'
else: p.write_text(json.dumps(payload,indent=2)+'\n')
PY
nice -n 10 .venv/bin/python scripts/experiments/er_next/run_gpu5_guarded.py \
  --states reports/er_next_20260916/fork_states.frozen.json \
  --checkpoint verl_checkpoints/opd-grpo-0.5B/actor/global_step_50 \
  --out "$RUN_DIR" --retriever-url http://127.0.0.1:18085/retrieve \
  --retriever-manifest "$RUN_DIR/retriever.frozen.json" \
  --limit "$LIMIT" --replicates "$REPLICATES"
if [[ "$MODE" != pilot ]]; then
  CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python \
    scripts/experiments/er_next/analyze_forks.py --out "$RUN_DIR"
fi
