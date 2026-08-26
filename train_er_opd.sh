#!/usr/bin/env bash
set -euo pipefail

# Standalone Evidence-Residual OPD + vanilla GRPO launcher. This script intentionally
# spells out the complete training configuration instead of delegating to
# train_opd.sh, so the ER experiment is self-contained and auditable.

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

# Formal-run values are deliberately fixed here.  Do not inherit stale exports
# from earlier smoke tests in the parent shell.  Explicit Hydra overrides passed
# through "$@" remain available for intentional one-off runs.
export CUDA_VISIBLE_DEVICES="2,3"
export VLLM_ATTENTION_BACKEND="XFORMERS"

DATA_DIR="data/nq_hotpotqa_train_30k_no_cold_start"
VAL_FILE="data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet"
TRAIN_DATA_SOURCE="hotpotqa"
VAL_DATA_SOURCE="null"
STUDENT_MODEL="data/student/1B"
TEACHER_MODEL="/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3"
EXPERIMENT_NAME="eropd-grpo-1B"
WAND_PROJECT="Search-R1-OPD2"
N_GPUS="2"
TOTAL_TRAINING_STEPS="201"
TRAIN_BATCH_SIZE="128"
VAL_BATCH_SIZE="512"
OPD_PPO_MINI_BATCH_SIZE="256"
OPD_PPO_MICRO_BATCH_SIZE="8"
OPD_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="32"
OPD_REF_LOG_PROB_MICRO_BATCH_SIZE="16"
OPD_LAMBDA_DISTILL="1.0"
OPD_GRPO_REWARD_COEF="1.0"
OPD_N_AGENT="8"
OPD_TARGET_TOKEN_CHUNK_SIZE="512"

OPD_DIAGNOSTICS_ENABLE="false"
OPD_DIAGNOSTICS_OUTPUT_DIR="null"
OPD_DIAGNOSTICS_EVERY_N_STEPS="1"
OPD_DIAGNOSTICS_MAX_SEQUENCES="64"
OPD_DIAGNOSTICS_SAMPLE_STRATEGY="random"
OPD_DIAGNOSTICS_COMPRESS="false"
OPD_ROLLOUT_SEED="null"
OPD_ENTROPY_CALIBRATION_ENABLE="false"
OPD_ENTROPY_CALIBRATION_FRACTION="0.05"
OPD_ENTROPY_CALIBRATION_SPLIT_SEED="42"
OPD_ENTROPY_CALIBRATION_DECODE_SEED="42"
OPD_ENTROPY_CALIBRATION_DATA_SOURCE="hotpotqa"
OPD_ENTROPY_CALIBRATION_OUTPUT="refine-logs/ER_ENTROPY_CALIBRATION.json"

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
    algorithm.adv_estimator=opd \
    algorithm.opd.advantage_mode=token \
    algorithm.opd.normalize=false \
    algorithm.opd.clip_value=null \
    algorithm.opd.distillation_coef=1.0 \
    algorithm.opd.mask_protocol_tags=false \
    algorithm.opd.lambda_distill="$OPD_LAMBDA_DISTILL" \
    algorithm.opd.teacher_target=evidence_residual \
    algorithm.opd.entropy_matched_tau=null \
    algorithm.opd.target_token_chunk_size="$OPD_TARGET_TOKEN_CHUNK_SIZE" \
    algorithm.opd.grpo_reward_coef="$OPD_GRPO_REWARD_COEF" \
    algorithm.opd.use_gated_distillation=false \
    algorithm.opd.gamma=1.0 \
    algorithm.opd.beta_min=0.0 \
    algorithm.opd.beta_max=0.05 \
    algorithm.opd.rce.enable=false \
    algorithm.opd.rce.entropy_normalization=percentile_rank \
    algorithm.opd.rce.w_min=0.1 \
    algorithm.opd.rce.w_max=1.0 \
    algorithm.opd.rce.alpha=4.0 \
    algorithm.opd.rce.tau=0.0 \
    algorithm.opd.rce.default_retrieval_hit=0.5 \
    algorithm.opd.diagnostics.enable="$OPD_DIAGNOSTICS_ENABLE" \
    algorithm.opd.diagnostics.output_dir="$OPD_DIAGNOSTICS_OUTPUT_DIR" \
    algorithm.opd.diagnostics.every_n_steps="$OPD_DIAGNOSTICS_EVERY_N_STEPS" \
    algorithm.opd.diagnostics.max_sequences_per_step="$OPD_DIAGNOSTICS_MAX_SEQUENCES" \
    algorithm.opd.diagnostics.sample_strategy="$OPD_DIAGNOSTICS_SAMPLE_STRATEGY" \
    algorithm.opd.diagnostics.compress="$OPD_DIAGNOSTICS_COMPRESS" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size="$OPD_PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$OPD_PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$OPD_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.seed="$OPD_ROLLOUT_SEED" \
    actor_rollout_ref.rollout.restrict_to_tokenizer_vocab=true \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent="$OPD_N_AGENT" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$OPD_REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
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
    trainer.er_entropy_calibration.enable="$OPD_ENTROPY_CALIBRATION_ENABLE" \
    trainer.er_entropy_calibration.fraction="$OPD_ENTROPY_CALIBRATION_FRACTION" \
    trainer.er_entropy_calibration.split_seed="$OPD_ENTROPY_CALIBRATION_SPLIT_SEED" \
    trainer.er_entropy_calibration.decode_seed="$OPD_ENTROPY_CALIBRATION_DECODE_SEED" \
    trainer.er_entropy_calibration.data_source="$OPD_ENTROPY_CALIBRATION_DATA_SOURCE" \
    trainer.er_entropy_calibration.output_path="$OPD_ENTROPY_CALIBRATION_OUTPUT" \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 "$@"
