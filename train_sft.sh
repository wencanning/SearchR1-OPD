#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
PYTHON_BIN=${PYTHON_BIN:-/data/home/wencanning/miniconda3/envs/searchr1/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export TRAIN_DATA=${TRAIN_DATA:-data/search_sft_7b_hotpotqa_8k/train.parquet}
export VAL_DATA=${VAL_DATA:-data/search_sft_7b_hotpotqa_8k/val.parquet}
export BASE_MODEL=${BASE_MODEL:-data/student/0.5B}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft-05B-teacher7B-hotpotqa-8k}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-verl_checkpoints/${EXPERIMENT_NAME}}
mkdir -p experiment_logs/sft
LOG_FILE=${LOG_FILE:-experiment_logs/sft/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=4096 \
    data.train_batch_size=64 \
    data.micro_batch_size=4 \
    model.partial_pretrain="${BASE_MODEL}" \
    model.enable_gradient_checkpointing=true \
    trainer.project_name=search-r1-sft \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    optim.lr=1e-5 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.default_hdfs_dir=null \
    "$@" 2>&1 | tee "${LOG_FILE}"
