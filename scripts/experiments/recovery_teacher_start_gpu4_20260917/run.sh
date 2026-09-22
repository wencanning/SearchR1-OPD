#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('reports/recovery_teacher_start_gpu4_20260917/witness_teacher/status.json')
s=json.loads(p.read_text())
assert s['state']=='complete' and s['completed']==s['total']==1
PY
.venv/bin/python scripts/experiments/recovery_teacher_start_gpu4_20260917/guard.py scripts/experiments/recovery_teacher_start_gpu4_20260917/recovery.py --phase question_start --model teacher --out reports/recovery_teacher_start_gpu4_20260917/question_start
.venv/bin/python scripts/experiments/recovery_teacher_start_gpu4_20260917/analyze.py
