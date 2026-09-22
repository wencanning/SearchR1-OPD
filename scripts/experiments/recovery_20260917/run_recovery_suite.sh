#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/recovery.py --phase collect --model student --out reports/recovery_20260917/collected
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/recovery.py --phase continue --model student --out reports/recovery_20260917/continue_student
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/recovery.py --phase continue --model teacher --out reports/recovery_20260917/continue_teacher
