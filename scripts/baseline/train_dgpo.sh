#!/usr/bin/env bash
set -euo pipefail

# Search-R1 adaptation of the official Agentic-RAG DGPO implementation:
#   https://github.com/omron-sinicx/dgpo (commit 33df0d0)
#
# This comparison substitutes our own 0.5B student and 7B Search-R1 teacher for
# the authors' released checkpoints. The online objective follows the official
# PPO/GAE implementation: correct trajectory -> EM reward only; incorrect
# trajectory -> selective forward-KL guidance from the teacher.
#
# Data, sequence lengths, learning rate, 200 rollout/update cycles, validation
# cadence, and two-A100 placement retain the formal 0.5B ER-OPD comparison
# setting e4fmtict. PPO-specific settings (GAE, one rollout per prompt, critic,
# warmup, entropy, and symmetric clipping) follow the official DGPO launcher.
# Values are fixed so stale parent-shell exports cannot alter the run; explicit
# Hydra overrides in "$@" remain available.

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES="2,3"
export VLLM_ATTENTION_BACKEND="XFORMERS"

DATA_DIR="data/nq_hotpotqa_train_30k_no_cold_start"
VAL_FILE="$DATA_DIR/validation_diagnostic_512.parquet"
TRAIN_DATA_SOURCE="hotpotqa"
VAL_DATA_SOURCE="null"
STUDENT_MODEL="data/student/0.5B"
TEACHER_MODEL="/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3"
EXPERIMENT_NAME="dgpo-ppo-0.5B-200steps"
WAND_PROJECT="Search-R1-OPD2"
N_GPUS="2"
# The legacy trainer starts at step 1 and stops after incrementing the counter;
# 201 therefore executes optimizer updates 1 through 200.
TOTAL_TRAINING_STEPS="201"
TRAIN_BATCH_SIZE="128"
VAL_BATCH_SIZE="512"
PPO_MINI_BATCH_SIZE="128"
ACTOR_PPO_MICRO_BATCH_SIZE="8"
CRITIC_PPO_MICRO_BATCH_SIZE="8"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="32"
REF_LOG_PROB_MICRO_BATCH_SIZE="16"
DGPO_REWARD_THRESHOLD="0.1"
DGPO_KL_COEF="0.001"
DGPO_N_AGENT="1"
DGPO_ROLLOUT_SEED="null"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$VAL_FILE" \
    data.train_data_source="$TRAIN_DATA_SOURCE" \
    data.val_data_source="$VAL_DATA_SOURCE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length=5120 \
    data.max_response_length=512 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=gae \
    algorithm.dgpo.enable=true \
    algorithm.dgpo.reward_threshold="$DGPO_REWARD_THRESHOLD" \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef="$DGPO_KL_COEF" \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.ref.model_path="$TEACHER_MODEL" \
    actor_rollout_ref.ref.attn_implementation=sdpa \
    actor_rollout_ref.ref.target_forward_dtype=float16 \
    actor_rollout_ref.ref.allow_approximate_target_precision=true \
    actor_rollout_ref.ref.target_select_policy_logits=true \
    actor_rollout_ref.ref.target_trim_shared_prompt_padding=true \
    actor_rollout_ref.ref.allow_tf32=false \
    actor_rollout_ref.ref.fsdp_config.model_dtype=float16 \
    actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=float16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$ACTOR_PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_type=kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.seed="$DGPO_ROLLOUT_SEED" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.restrict_to_tokenizer_vocab=true \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent="$DGPO_N_AGENT" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    critic.model.path="$STUDENT_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=true \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    critic.ppo_micro_batch_size="$CRITIC_PPO_MICRO_BATCH_SIZE" \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.grad_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=false \
    +trainer.val_only=false \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=15 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 "$@"
