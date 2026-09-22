#!/usr/bin/env bash
# Matched causal screen from one frozen step75 student; fresh optimizer in both arms.
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
GPU="${1:?GPU required}"
RUN_ID="${2:?run id required}"
MASK="${3:?true or false}"
[[ "$GPU" =~ ^[23]$ && "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$MASK" =~ ^(true|false)$ ]] || exit 2
test ! -e "verl_checkpoints/pure-opd-05B-n1-$RUN_ID" || { echo 'Refusing to overwrite an existing run'; exit 2; }
bash scripts/baseline/train_pure_opd_sod.sh opd "$GPU" "$RUN_ID" \
  actor_rollout_ref.model.path=verl_checkpoints/pure-opd-05B-n1-20260914-1220-coef1/actor/global_step_75 \
  algorithm.opd.mask_protocol_tags="$MASK" \
  data.train_seed=42 actor_rollout_ref.rollout.seed=42 \
  trainer.logger='[console]' trainer.total_training_steps=41 trainer.save_freq=5 trainer.test_freq=5
