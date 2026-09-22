#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
exec 9>reports/strict_role_cpu_20260916/run.lock
flock -n 9
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=.
trap 'rc=$?; if (( rc != 0 )); then echo "STRICT_ROLE_FAILED exit=$rc"; fi' EXIT
nice -n 10 .venv/bin/python scripts/experiments/action_calibration_cpu/verify_execution.py
nice -n 10 .venv/bin/python scripts/experiments/strict_role_cpu/local_update.py update
nice -n 10 .venv/bin/python scripts/experiments/action_calibration_cpu/verify_execution.py
nice -n 10 .venv/bin/python scripts/experiments/strict_role_cpu/analyze.py
