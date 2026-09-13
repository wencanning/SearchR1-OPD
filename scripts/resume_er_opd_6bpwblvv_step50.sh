#!/usr/bin/env bash
set -euo pipefail
cd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
source /data/home/wencanning/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
checkpoint="$PWD/verl_checkpoints/eropd-grpo-05B-test-alpha05/actor/global_step_50"
test -s "$checkpoint/model.safetensors"
stamp=$(date +%Y%m%d_%H%M%S)
experiment="eropd-alpha05-from-6bpwblvv-step50-$stamp"
export WANDB_RUN_ID="$(python -c 'import wandb; print(wandb.util.generate_id())')"
export WANDB_RESUME=never
export WANDB_RUN_GROUP=6bpwblvv-step50-continuation
export WANDB_NOTES="Weight-only continuation of 6bpwblvv at step 50. Optimizer, scheduler, RNG and dataloader restarted; first update is step 51. Original history is preserved."
export WANDB_TAGS=weight-only-continuation,parent-6bpwblvv,checkpoint-step50
mkdir -p experiment_logs/er_opd
printf 'New run: %s; experiment: %s; source: %s\n' "$WANDB_RUN_ID" "$experiment" "$checkpoint"
bash train_er_opd.sh \
  actor_rollout_ref.model.path="$checkpoint" \
  trainer.experiment_name="$experiment" \
  trainer.default_local_dir="verl_checkpoints/$experiment" \
  +trainer.warm_start_step=50 \
  +trainer.parent_run_id=6bpwblvv \
  +trainer.continuation_mode=weights_only \
  "$@" 2>&1 | tee "experiment_logs/er_opd/$experiment.log"
