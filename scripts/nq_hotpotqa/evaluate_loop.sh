#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=verl_checkpoints/eropd-grpo-0.5B/actor/global_step_150

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_DIR=data/nq_hotpotqa_train
VAL_FILE="$DATA_DIR/test.parquet"
RETRIEVER_URL=http://127.0.0.1:8000/retrieve
MODEL_LOG_NAME="$(python3 - "$MODEL_PATH" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
parts = path.parts
if len(parts) >= 4 and parts[-2] == "actor" and parts[-1].startswith("global_step_"):
    print(parts[-3])
else:
    print(path.name)
PY
)"
LOG_DIR="eval_loop_logs/$MODEL_LOG_NAME"
SLEEP_SECONDS=0
N_GPUS="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
EVAL_DO_SAMPLE=true
EVAL_SEED_BASE=""
STOP_REQUESTED=0
CURRENT_EVAL_PID=""

mkdir -p "$LOG_DIR/raw"

LATEST_RESULT_LOG="$LOG_DIR/latest_result.log"
BEST_RESULT_LOG="$LOG_DIR/best_result.log"
SUMMARY_LOG="$LOG_DIR/summary.tsv"

request_stop() {
    if [[ "$STOP_REQUESTED" == "1" ]]; then
        echo "[eval-loop] stop requested again; exiting"
        exit 130
    fi

    STOP_REQUESTED=1
    echo
    echo "[eval-loop] stop requested; interrupting current evaluation"
    if [[ -n "$CURRENT_EVAL_PID" ]] && kill -0 "$CURRENT_EVAL_PID" 2>/dev/null; then
        kill -INT "$CURRENT_EVAL_PID" 2>/dev/null || true
    fi
}

trap request_stop INT TERM

if [[ ! -f "$SUMMARY_LOG" ]]; then
    printf "iteration\ttimestamp\texit_code\tscore\traw_log\tmetric_line\n" > "$SUMMARY_LOG"
fi

extract_metric_line() {
    local raw_log="$1"
    grep -a "val/test_score/" "$raw_log" | tail -n 1 || true
}

extract_score() {
    local metric_line="$1"
    python3 - "$metric_line" <<'PY'
import re
import sys

line = sys.argv[1]
metric_pairs = re.findall(
    r"['\"]?(val/test_score/[A-Za-z0-9_/-]+)['\"]?\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    line,
)
metrics = {name: float(value) for name, value in metric_pairs}

if "val/test_score/Avg" in metrics:
    print(metrics["val/test_score/Avg"])
    raise SystemExit(0)

datasets = [
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "bamboogle",
]
keys = [f"val/test_score/{dataset}" for dataset in datasets]
if all(key in metrics for key in keys):
    print(sum(metrics[key] for key in keys) / len(keys))
PY
}

is_better_score() {
    local candidate="$1"
    local current_best="$2"
    python3 - "$candidate" "$current_best" <<'PY'
import sys

candidate = float(sys.argv[1])
current_best = sys.argv[2]
if current_best == "":
    raise SystemExit(0)
raise SystemExit(0 if candidate > float(current_best) else 1)
PY
}

run_eval_once() {
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files="$DATA_DIR/train.parquet" \
        data.val_files="$VAL_FILE" \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=512 \
        data.val_batch_size=512 \
        data.max_prompt_length=4096 \
        data.max_response_length=500 \
        data.max_start_length=2048 \
        data.max_obs_length=500 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=gae \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size=64 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.grad_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.n_agent=1 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.seed="$EVAL_SEED" \
        actor_rollout_ref.rollout.val_do_sample="$EVAL_DO_SAMPLE" \
        actor_rollout_ref.actor.state_masking=true \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.optim.lr_warmup_steps_ratio=0.05 \
        critic.model.path="$MODEL_PATH" \
        critic.model.enable_gradient_checkpointing=true \
        critic.ppo_micro_batch_size=8 \
        critic.model.fsdp_config.param_offload=true \
        critic.model.fsdp_config.grad_offload=true \
        critic.model.fsdp_config.optimizer_offload=true \
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.no_think_rl=false \
        trainer.critic_warmup=0 \
        trainer.logger=[] \
        +trainer.val_only=true \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node="$N_GPUS" \
        trainer.nnodes=1 \
        max_turns=4 \
        retriever.url="$RETRIEVER_URL" \
        retriever.topk=3
}

best_score=""
if [[ -f "$BEST_RESULT_LOG" ]]; then
    best_score="$(grep -a "^score=" "$BEST_RESULT_LOG" | tail -n 1 | cut -d= -f2- || true)"
fi

iteration=0
while true; do
    iteration=$((iteration + 1))
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    safe_timestamp="$(date '+%Y%m%d_%H%M%S')"
    raw_log="$LOG_DIR/raw/eval_${iteration}_${safe_timestamp}.log"
    if [[ -n "$EVAL_SEED_BASE" ]]; then
        EVAL_SEED=$((EVAL_SEED_BASE + iteration - 1))
    else
        EVAL_SEED="$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')"
    fi

    echo "[eval-loop] iteration=$iteration timestamp=$timestamp model=$MODEL_PATH sample=$EVAL_DO_SAMPLE seed=$EVAL_SEED"

    set +e
    run_eval_once > >(tee "$raw_log") 2>&1 &
    CURRENT_EVAL_PID=$!
    wait "$CURRENT_EVAL_PID"
    exit_code=$?
    CURRENT_EVAL_PID=""
    set -e

    if [[ "$STOP_REQUESTED" == "1" ]]; then
        echo "[eval-loop] stopped by user"
        exit 130
    fi

    metric_line="$(extract_metric_line "$raw_log")"
    score=""
    if [[ -n "$metric_line" ]]; then
        score="$(extract_score "$metric_line")"
    fi

    {
        echo "iteration=$iteration"
        echo "timestamp=$timestamp"
        echo "exit_code=$exit_code"
        echo "score=$score"
        echo "raw_log=$raw_log"
        echo "metric_line=$metric_line"
    } > "$LATEST_RESULT_LOG"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$iteration" "$timestamp" "$exit_code" "$score" "$raw_log" "$metric_line" >> "$SUMMARY_LOG"

    if [[ -n "$score" ]] && is_better_score "$score" "$best_score"; then
        best_score="$score"
        {
            echo "iteration=$iteration"
            echo "timestamp=$timestamp"
            echo "score=$score"
            echo "raw_log=$raw_log"
            echo "metric_line=$metric_line"
        } > "$BEST_RESULT_LOG"
        echo "[eval-loop] new best score=$score"
    fi

    if [[ "$exit_code" -ne 0 ]]; then
        echo "[eval-loop] evaluation failed with exit_code=$exit_code; continuing"
    fi

    if [[ "$STOP_REQUESTED" == "1" ]]; then
        echo "[eval-loop] stopped by user"
        exit 130
    fi

    if [[ "$SLEEP_SECONDS" != "0" ]]; then
        sleep "$SLEEP_SECONDS"
    fi
done
