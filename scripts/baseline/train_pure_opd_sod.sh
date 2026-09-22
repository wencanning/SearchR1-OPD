#!/usr/bin/env bash
# Matched single-GPU, single-rollout OPD / SOD-without-GRPO experiment.
set -euo pipefail
METHOD="${1:?usage: bash scripts/baseline/train_pure_opd_sod.sh opd|sod GPU RUN_TAG [Hydra overrides]}"
GPU="${2:?physical GPU required}"
RUN_TAG="${3:?unique paired run tag required}"
shift 3
case "$METHOD" in
  opd) SOD=false ;;
  sod) SOD=true ;;
  *) exit 2 ;;
esac
[[ "$GPU" =~ ^[0-9]+$ && "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="/data/home/wencanning/miniconda3/envs/searchr1/bin:$PATH"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONHASHSEED=42
export SEARCHR1_RAY_CPUS=16
export RAY_TMPDIR="$(mktemp -d /tmp/sr1-ray-XXXXXXXX)"
export WANDB_RUN_ID="${RUN_TAG}-${METHOD}"
export WANDB_RESUME=never
NAME="pure-${METHOD}-05B-n1-${RUN_TAG}"
OUT="verl_checkpoints/$NAME"
mkdir -p "$RAY_TMPDIR" "$OUT"
python3 -m verl.trainer.main_ppo \
  data.train_files=data/nq_hotpotqa_train_30k_no_cold_start/train.parquet \
  data.val_files=data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet \
  data.train_data_source=hotpotqa data.val_data_source=null \
  data.train_batch_size=128 data.val_batch_size=128 data.train_seed=42 \
  data.max_prompt_length=5120 data.max_response_length=512 \
  data.max_start_length=2048 data.max_obs_length=512 \
  algorithm.adv_estimator=opd algorithm.opd.advantage_mode=token \
  algorithm.opd.normalize=false algorithm.opd.clip_value=null \
  algorithm.opd.distillation_coef=1.0 algorithm.opd.lambda_distill=1.0 \
  algorithm.opd.teacher_target=observed algorithm.opd.grpo_reward_coef=0.0 \
  algorithm.opd.mask_protocol_tags=false algorithm.opd.use_gated_distillation=false \
  algorithm.opd.rce.enable=false algorithm.opd.sod.enable="$SOD" \
  algorithm.opd.sod.allow_no_grpo_ablation="$SOD" \
  algorithm.opd.sod.epsilon=1e-6 algorithm.opd.sod.delta=0.2 \
  algorithm.opd.target_token_chunk_size=256 algorithm.no_think_rl=false \
  algorithm.opd.diagnostics.enable=true algorithm.opd.diagnostics.output_dir="$OUT/opd_diagnostics" \
  algorithm.opd.diagnostics.every_n_steps=1 algorithm.opd.diagnostics.max_sequences_per_step=128 \
  algorithm.opd.diagnostics.sample_strategy=random algorithm.opd.diagnostics.compress=true \
  algorithm.opd.diagnostics.include_token_text=true algorithm.opd.diagnostics.include_decoded_response=true \
  algorithm.opd.diagnostics.float_precision=6 \
  actor_rollout_ref.model.path=data/student/0.5B \
  actor_rollout_ref.ref.model_path=/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.ref.attn_implementation=sdpa actor_rollout_ref.ref.observed_target_backend=padded \
  actor_rollout_ref.ref.target_forward_dtype=float16 actor_rollout_ref.ref.allow_approximate_target_precision=true \
  actor_rollout_ref.ref.target_select_policy_logits=true actor_rollout_ref.ref.target_trim_shared_prompt_padding=true \
  actor_rollout_ref.ref.allow_tf32=false actor_rollout_ref.ref.fsdp_config.model_dtype=float16 \
  actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=float16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=2 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=3.0 actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0.0 actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.restrict_to_tokenizer_vocab=true \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.n_agent=1 \
  trainer.logger='[console,wandb]' +trainer.val_before_train=false +trainer.val_only=false \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=25 trainer.test_freq=10 \
  trainer.project_name=Search-R1-OPD2 trainer.experiment_name="$NAME" \
  trainer.total_epochs=15 trainer.total_training_steps=151 \
  trainer.default_hdfs_dir=null trainer.default_local_dir="$OUT" \
  trainer.er_entropy_calibration.enable=false \
  max_turns=4 retriever.url=http://127.0.0.1:8000/retrieve retriever.topk=3 "$@" \
  2>&1 | tee "$OUT/train.log"
