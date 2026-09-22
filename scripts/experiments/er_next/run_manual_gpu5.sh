#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-help}"
RETRIEVER_URL="${2:-}"
if [[ "$MODE" != pilot && "$MODE" != full ]] || [[ -z "$RETRIEVER_URL" ]]; then
  echo 'Usage: bash scripts/experiments/er_next/run_manual_gpu5.sh {pilot|full} RETRIEVER_URL'
  echo 'pilot: 2 states x 2 actions x 1 sample; full: 64 states x 2 actions x 4 samples.'
  echo 'No retriever or training process is started. Only the 0.5B student runs on guarded GPU5.'
  exit 2
fi
RUN_DIR="reports/er_next_20260916/forks_gpu5_${MODE}"
LIMIT=0
REPLICATES=4
if [[ "$MODE" == pilot ]]; then
  LIMIT=2
  REPLICATES=1
fi
nice -n 10 .venv/bin/python scripts/experiments/er_next/run_gpu5_guarded.py \
  --states reports/er_next_20260916/fork_states.frozen.json \
  --checkpoint verl_checkpoints/opd-grpo-0.5B/actor/global_step_50 \
  --out "$RUN_DIR" \
  --retrieval-cache reports/evidence_stopping_gpu5_20260915/retrieval_cache.json \
  --retriever-url "$RETRIEVER_URL" --limit "$LIMIT" --replicates "$REPLICATES"
if [[ "$MODE" == full ]]; then
  CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python \
    scripts/experiments/er_next/analyze_forks.py --out "$RUN_DIR"
fi
