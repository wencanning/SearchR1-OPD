#!/usr/bin/env bash
set -uo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
.venv/bin/python scripts/experiments/recovery_continue_gpu4_20260917/guard.py scripts/experiments/recovery_continue_gpu4_20260917/recovery.py --phase continue --model student --out reports/recovery_continue_gpu4_20260917/continue_student
collection_exit=$?
.venv/bin/python scripts/experiments/recovery_continue_gpu4_20260917/finish_handoff.py --collect-exit "$collection_exit"
handoff_exit=$?
if [ "$handoff_exit" -ne 0 ]; then
    exit "$handoff_exit"
fi
exit "$collection_exit"
