#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('reports/recovery_20260917')
for name in ['continue_student','continue_teacher','witness_training']:
    assert json.loads((p/name/'status.json').read_text())['state']=='complete',name
s=json.loads((p/'recovery_summary.json').read_text())
assert s['teacher_rescued_questions']>=4 and s['verified_student_suffixes']>=4,'Pilot qualification failed'
PY
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/score_targets.py --out reports/recovery_20260917/teacher_scores
for arm in base student_replay teacher_repair; do
    .venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/local_update.py --arm "$arm" --out "reports/recovery_20260917/update_$arm"
    if [ "$arm" = base ]; then
        .venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/evaluate_update.py --phase continue --model student --checkpoint reports/recovery_20260917/update_base/zero_step_checkpoint --out reports/recovery_20260917/eval_zero
    fi
    .venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/evaluate_update.py --phase continue --model student --checkpoint "reports/recovery_20260917/update_$arm/checkpoint" --out "reports/recovery_20260917/eval_$arm"
done
.venv/bin/python scripts/experiments/recovery_20260917/analyze_update.py
