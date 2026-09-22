#!/usr/bin/env bash
# Explicit GPU3 work queue. A STOP file prevents the next job; no restart storm.
set -uo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
OUT=reports/search_loop_20260914
PY=/data/home/wencanning/miniconda3/envs/searchr1/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=3
"$PY" scripts/diagnostics/probe_search_loop.py 2>&1 | tee -a "$OUT/gpu3.log"
DIAG_STATUS=${PIPESTATUS[0]}
echo "Diagnostic exited: $DIAG_STATUS"
if test -f "$OUT/DELEGATE_GPU3"; then exit "$DIAG_STATUS"; fi
for DIAG_SEED in 43 44 45; do
  if test -f "$OUT/STOP"; then exit 0; fi
  "$PY" -c 'from scripts.diagnostics.probe_search_loop import gpu_check; gpu_check()' || exit 75
  DIAG_START=$SECONDS
  setsid bash scripts/baseline/train_pure_opd_sod.sh opd 3 "searchloop-control-seed${DIAG_SEED}-20260914" \
    data.train_seed="$DIAG_SEED" actor_rollout_ref.rollout.seed="$DIAG_SEED" \
    trainer.logger='[console]' trainer.total_training_steps=151 \
    trainer.save_freq=10 &
  DIAG_JOB_PID=$!
  wait "$DIAG_JOB_PID"
  DIAG_STATUS=$?
  # Only the dedicated control's process group, never the shared retriever or queue.
  "$PY" - "$DIAG_JOB_PID" <<'PY'
import os,signal,sys,time
group=int(sys.argv[1])
assert group>1 and group!=os.getpgrp() and group!=os.getpgid(110019)
try:os.killpg(group,signal.SIGTERM)
except ProcessLookupError:pass
time.sleep(2)
PY
  echo "Control seed $DIAG_SEED exited: $DIAG_STATUS, elapsed $((SECONDS-DIAG_START)) s"
  if test "$DIAG_STATUS" -ne 0; then exit "$DIAG_STATUS"; fi
done
