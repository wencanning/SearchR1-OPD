#!/usr/bin/env bash
# User-requested useful-work fallback on ONE card; never touches its peer.
set -euo pipefail
GPU="${1:?GPU}" NAME="${2:?unique experiment name}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="/data/home/wencanning/miniconda3/envs/searchr1/bin:$PATH"
export ER_OPD_CUDA_VISIBLE_DEVICES="$GPU"
export SEARCHR1_RAY_CPUS=16
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export RAY_TMPDIR="/tmp/$NAME"
export WANDB_RUN_ID="$NAME" WANDB_RESUME=never
mkdir -p "$RAY_TMPDIR" "verl_checkpoints/$NAME"
bash train_er_opd.sh \
  trainer.n_gpus_per_node=1 trainer.experiment_name="$NAME" \
  trainer.default_local_dir="verl_checkpoints/$NAME" \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
  algorithm.opd.target_token_chunk_size=128 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.free_cache_engine=false \
  2>&1 | tee "verl_checkpoints/$NAME/train.log"
