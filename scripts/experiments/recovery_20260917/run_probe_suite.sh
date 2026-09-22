#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/probe.py --model student --out reports/recovery_20260917/probe_student
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/probe.py --model teacher --out reports/recovery_20260917/probe_teacher
