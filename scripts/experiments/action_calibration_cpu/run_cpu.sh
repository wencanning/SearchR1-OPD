#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
mkdir -p reports/action_calibration_cpu_20260916
exec 9>reports/action_calibration_cpu_20260916/run.lock
flock -n 9 || { echo 'Action-calibration CPU experiment already running'; exit 1; }
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
run_stage() {
    nice -n 10 .venv/bin/python scripts/experiments/action_calibration_cpu/verify_execution.py
    nice -n 10 .venv/bin/python "$@"
}
run_stage scripts/experiments/action_calibration_cpu/score_bias.py calibration
run_stage scripts/experiments/action_calibration_cpu/analyze_bias.py calibrate
run_stage scripts/experiments/action_calibration_cpu/score_bias.py test
run_stage scripts/experiments/action_calibration_cpu/analyze_bias.py summarize
run_stage scripts/experiments/action_calibration_cpu/local_update.py sample
run_stage scripts/experiments/action_calibration_cpu/local_update.py teacher
run_stage scripts/experiments/action_calibration_cpu/local_update.py update
run_stage scripts/experiments/action_calibration_cpu/analyze_update.py
