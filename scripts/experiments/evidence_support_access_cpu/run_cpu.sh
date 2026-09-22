#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
mkdir -p reports/evidence_support_access_cpu_20260916
exec 9>reports/evidence_support_access_cpu_20260916/run.lock
flock -n 9 || { echo 'Support-access experiment already running'; exit 1; }
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
nice -n 10 .venv/bin/python scripts/experiments/evidence_support_access_cpu/worker.py run
nice -n 10 .venv/bin/python scripts/experiments/evidence_support_access_cpu/summarize.py
