#!/usr/bin/env bash
# Reuse the existing GPU2/3 retriever; only start a guarded GPU5 student.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${1:-reports/er_next_20260916/forks_shared_gpu23_full}"
[[ -f "$RUN_DIR/retriever.frozen.json" ]] || { echo 'Missing frozen shared retriever preflight'; exit 2; }
finish() {
  EXIT_CODE=$?
  CUDA_VISIBLE_DEVICES='' .venv/bin/python - "$RUN_DIR" "$EXIT_CODE" <<'PY'
import datetime,json,sys
from pathlib import Path
out=Path(sys.argv[1]);code=int(sys.argv[2])
(out/'launcher_status.json').write_text(json.dumps(dict(
    state='complete' if code==0 else 'failed',exit_code=code,
    finished_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
PY
}
trap finish EXIT
nice -n 10 .venv/bin/python scripts/experiments/er_next/run_gpu5_guarded.py \
  --states reports/er_next_20260916/fork_states.frozen.json \
  --checkpoint verl_checkpoints/opd-grpo-0.5B/actor/global_step_50 \
  --out "$RUN_DIR" --retriever-url http://127.0.0.1:8000/retrieve \
  --retriever-manifest "$RUN_DIR/retriever.frozen.json" --limit 0 --replicates 4
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/experiments/er_next/analyze_forks.py --out "$RUN_DIR"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python \
  scripts/experiments/er_next/summarize_independent.py --out "$RUN_DIR"
