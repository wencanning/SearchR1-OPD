#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
GPU="${1:?GPU required}"
RUN_ID="${2:?unique run id required}"
[[ "$GPU" =~ ^[23]$ && "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p "reports/$RUN_ID"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
/data/home/wencanning/miniconda3/envs/searchr1/bin/python scripts/diagnostics/probe_action_credit.py \
  --gpu "$GPU" --run-id "$RUN_ID" 2>&1 | tee "reports/$RUN_ID/run.log"
