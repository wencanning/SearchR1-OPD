#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8001/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2}"
TEACHER_TOKENIZER="${TEACHER_TOKENIZER:-Qwen/Qwen2.5-3B-Instruct}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

DATA_SOURCES="${DATA_SOURCES:-nq,hotpotqa}"
SPLIT="${SPLIT:-train}"
SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-5000}"
SEED="${SEED:-7}"
CONCURRENCY="${CONCURRENCY:-24}"
TOPK="${TOPK:-3}"
MAX_TURNS="${MAX_TURNS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
MAX_INVALID_ACTIONS="${MAX_INVALID_ACTIONS:-2}"
MIN_SEARCH_TURNS="${MIN_SEARCH_TURNS:-1}"
MAX_SEARCH_TURNS="${MAX_SEARCH_TURNS:-4}"
VAL_RATIO="${VAL_RATIO:-0.02}"

ROLLOUT_DIR="${ROLLOUT_DIR:-data/teacher_rollout}"
PARQUET_DIR="${PARQUET_DIR:-data/search_sft}"
REQUIRE_RETRIEVAL_HIT="${REQUIRE_RETRIEVAL_HIT:-0}"
OVERWRITE_ROLLOUT="${OVERWRITE_ROLLOUT:-0}"

ROLLOUT_ARGS=(
  scripts/sft/rollout_teacher_vllm.py
  --base-url "${TEACHER_BASE_URL}"
  --model "${TEACHER_MODEL}"
  --tokenizer "${TEACHER_TOKENIZER}"
  --retriever-url "${RETRIEVER_URL}"
  --data-sources "${DATA_SOURCES}"
  --split "${SPLIT}"
  --samples-per-source "${SAMPLES_PER_SOURCE}"
  --seed "${SEED}"
  --output-dir "${ROLLOUT_DIR}"
  --concurrency "${CONCURRENCY}"
  --topk "${TOPK}"
  --max-turns "${MAX_TURNS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --max-invalid-actions "${MAX_INVALID_ACTIONS}"
  --min-search-turns "${MIN_SEARCH_TURNS}"
  --max-search-turns "${MAX_SEARCH_TURNS}"
)

if [[ "${REQUIRE_RETRIEVAL_HIT}" == "1" ]]; then
  ROLLOUT_ARGS+=(--require-retrieval-hit)
fi

if [[ "${OVERWRITE_ROLLOUT}" == "1" ]]; then
  ROLLOUT_ARGS+=(--overwrite)
fi

echo "[1/2] Teacher rollout -> ${ROLLOUT_DIR}"
"${PYTHON_BIN}" "${ROLLOUT_ARGS[@]}"

echo "[2/2] Accepted rollout -> parquet -> ${PARQUET_DIR}"
"${PYTHON_BIN}" scripts/sft/raw_to_parquet.py \
  --input-jsonl "${ROLLOUT_DIR}/accepted_rollouts.jsonl" \
  --output-dir "${PARQUET_DIR}" \
  --val-ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --require-quality-pass

echo "Done."
echo "Raw rollout: ${ROLLOUT_DIR}/raw_rollouts.jsonl"
echo "Accepted rollout: ${ROLLOUT_DIR}/accepted_rollouts.jsonl"
echo "Train parquet: ${PARQUET_DIR}/train.parquet"
echo "Val parquet: ${PARQUET_DIR}/val.parquet"
