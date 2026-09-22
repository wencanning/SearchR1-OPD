#!/usr/bin/env bash
# Own CPU service lifecycle; existing GPU2/3 retrieval/training is never contacted.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-help}"
case "$MODE" in
  bounded|full|pilot) ;;
  *) echo 'Usage: bash scripts/experiments/er_next/launch_independent.sh bounded|pilot|full'; exit 2 ;;
esac
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import socket
with socket.socket() as s:
    s.bind(('127.0.0.1',18085))
PY
SERVICE_DIR="reports/er_next_20260916/cpu_service_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$SERVICE_DIR"
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 /data/home/wencanning/miniconda3/envs/retriever/bin/python \
  scripts/experiments/er_next/cpu_retriever.py --out "$SERVICE_DIR" --port 18085 \
  > "$SERVICE_DIR/server.log" 2>&1 &
CPU_PID=$!
cleanup() {
  if kill -0 "$CPU_PID" 2>/dev/null; then
    kill -TERM "$CPU_PID"
    wait "$CPU_PID" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
echo "CPU_SERVICE_PID=$CPU_PID LOG=$SERVICE_DIR/server.log"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python - "$CPU_PID" <<'PY'
import os,sys,time,requests
pid=int(sys.argv[1]); session=requests.Session(); session.trust_env=False
for _ in range(450):
    os.kill(pid,0)
    try:
        r=session.get('http://127.0.0.1:18085/health',timeout=2)
        r.raise_for_status()
        assert r.json()['pid']==pid, 'Port belongs to another service'
        print('INDEPENDENT_CPU_SERVICE_READY',flush=True)
        break
    except requests.RequestException:
        time.sleep(2)
else: raise TimeoutError('Independent CPU service not ready within bounded startup wait')
PY
if [[ "$MODE" == bounded ]]; then
  bash scripts/experiments/er_next/run_independent_forks.sh pilot
  bash scripts/experiments/er_next/run_independent_forks.sh small32
else
  bash scripts/experiments/er_next/run_independent_forks.sh "$MODE"
fi
