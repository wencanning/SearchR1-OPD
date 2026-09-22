#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('reports/recovery_20260917_native')
for name in ['continue_student','continue_teacher']:
    assert json.loads((p/name/'status.json').read_text())['state']=='complete',name
assert json.loads((p.parent/'recovery_20260917/witness_training/status.json').read_text())['state']=='complete','Unchanged compute environment backward witness'
s=json.loads((p/'recovery_summary.json').read_text())
assert s['teacher_rescued_questions']>=4 and s['verified_student_suffixes']>=4,'Pilot qualification failed'
PY
.venv/bin/python scripts/experiments/recovery_20260917_native/guard.py scripts/experiments/recovery_20260917_native/score_targets.py --out reports/recovery_20260917_native/teacher_scores
for arm in base student_replay teacher_repair; do
    .venv/bin/python scripts/experiments/recovery_20260917_native/guard.py scripts/experiments/recovery_20260917_native/local_update.py --arm "$arm" --out "reports/recovery_20260917_native/update_$arm"
    if [ "$arm" = base ]; then
        .venv/bin/python scripts/experiments/recovery_20260917_native/verify_baseline_reuse.py
    fi
    .venv/bin/python scripts/experiments/recovery_20260917_native/guard.py scripts/experiments/recovery_20260917_native/evaluate_update.py --phase continue --model student --checkpoint "reports/recovery_20260917_native/update_$arm/checkpoint" --out "reports/recovery_20260917_native/eval_$arm"
done
.venv/bin/python scripts/experiments/recovery_20260917_native/analyze_update.py
