#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ENTROPY_MATCHED_TAU:-}" ]]; then
    echo "ENTROPY_MATCHED_TAU must be set to the pre-calibrated global temperature" >&2
    exit 2
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-hotpotqa-search-r1-entropy-matched-opd}"
export TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-hotpotqa}"
export VAL_DATA_SOURCE="${VAL_DATA_SOURCE:-hotpotqa}"
export OPD_TEACHER_TARGET=entropy_matched
export OPD_ENTROPY_MATCHED_TAU="$ENTROPY_MATCHED_TAU"
export OPD_LAMBDA_DISTILL="${OPD_LAMBDA_DISTILL:-1.0}"
export OPD_GRPO_REWARD_COEF=0.0
export OPD_USE_GATED_DISTILLATION=false
export OPD_RCE_ENABLE=false
export OPD_N_AGENT=1
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export OPD_PPO_MINI_BATCH_SIZE="$TRAIN_BATCH_SIZE"
export OPD_REF_ATTN_IMPLEMENTATION=sdpa
export OPD_REF_MODEL_DTYPE=fp32
export OPD_REF_PARAM_DTYPE=fp32
export OPD_REF_TARGET_FORWARD_DTYPE=fp32
CONTROL_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-4,5}"
CONTROL_GPU_COUNT="${N_GPUS:-$(awk -F, '{print NF}' <<< "$CONTROL_VISIBLE_GPUS")}"
export OPD_REF_LOG_PROB_MICRO_BATCH_SIZE="${OPD_REF_LOG_PROB_MICRO_BATCH_SIZE:-$CONTROL_GPU_COUNT}"
export OPD_REF_LOG_PROB_USE_DYNAMIC_BSZ=false
export OPD_TARGET_TOKEN_CHUNK_SIZE="${OPD_TARGET_TOKEN_CHUNK_SIZE:-16}"
export OPD_RESTRICT_TO_TOKENIZER_VOCAB=true

exec bash "$(dirname "$0")/train_opd.sh" "$@"
