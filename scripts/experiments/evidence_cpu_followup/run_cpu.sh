#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
exec 9>reports/evidence_cpu_followup_20260915/run.lock
flock -n 9 || { echo 'CPU follow-up already running'; exit 1; }
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
nice -n 10 .venv/bin/python scripts/experiments/evidence_cpu_followup/worker.py all
nice -n 10 .venv/bin/python scripts/experiments/evidence_cpu_followup/summarize.py
