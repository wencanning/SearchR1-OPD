#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
exec 9>reports/evidence_use_gpu5_20260915/run.lock
flock -n 9 || { echo 'This experiment is already running'; exit 1; }
export CUDA_VISIBLE_DEVICES=GPU-2abc6681-2116-879c-a361-749edb32ac3b
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
import os, subprocess
gpu=os.environ['CUDA_VISIBLE_DEVICES']
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
row=next(r.split(', ') for r in rows.splitlines() if gpu in r)
assert row[0]=='5' and int(row[2])<500, 'GPU 5 is occupied; no existing processes will be stopped'
p=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert gpu not in p, 'GPU 5 has an existing compute process'
print('PREFLIGHT_GPU5', ', '.join(row),flush=True)
PY
.venv/bin/python scripts/experiments/evidence_use/experiment.py run
