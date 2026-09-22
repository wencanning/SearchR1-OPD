#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
exec 9>reports/evidence_stopping_gpu5_20260915/run.lock
flock -n 9 || { echo 'Follow-up is already running'; exit 1; }
export CUDA_VISIBLE_DEVICES=GPU-2abc6681-2116-879c-a361-749edb32ac3b
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
import subprocess,os
gpu=os.environ['CUDA_VISIBLE_DEVICES']
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
r=next(s.split(', ') for s in rows.splitlines() if gpu in s)
assert r[0]=='5' and int(r[2])<500,'GPU 5 occupied; existing jobs will not be stopped'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert gpu not in apps,'GPU 5 has an existing compute process'
print('PREFLIGHT_GPU5',', '.join(r),flush=True)
PY
.venv/bin/python scripts/experiments/evidence_stopping/run.py --smoke
.venv/bin/python scripts/experiments/evidence_stopping/run.py
.venv/bin/python scripts/experiments/evidence_stopping/analyze.py
