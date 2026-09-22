#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONUNBUFFERED=1
nice -n 10 .venv/bin/python scripts/experiments/action_calibration_cpu/verify_execution.py
nice -n 10 .venv/bin/python scripts/experiments/er_generation_cpu/run.py pilot
nice -n 10 .venv/bin/python scripts/experiments/action_calibration_cpu/verify_execution.py
nice -n 10 .venv/bin/python scripts/experiments/er_generation_cpu/analyze.py pilot
