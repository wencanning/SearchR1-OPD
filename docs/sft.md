# Search-R1 SFT

This repository now includes a masked SFT path for distilling Search-R1 trajectories into a smaller student model.

## Expected dataset schema

Training data should be stored as parquet files with at least these columns:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Answer the given question..."
    }
  ],
  "response": "<think>...</think><search>...</search><information>...</information><answer>...</answer>"
}
```

`prompt` follows the same chat-style format used by the RL pipeline. `response` should contain the full teacher trajectory.

## Loss masking

The SFT dataset masks out:

- every prompt token
- every `<information>...</information>` span

The remaining response tokens, such as `<think>`, `<search>`, and `<answer>`, contribute to the training loss.

## Local development vs. server training

The local `uv` environment is intended for code editing and CPU unit tests. It is deliberately lighter than the full training environment.

For actual A100 training, keep using the server-side CUDA environment and install the full project dependencies there.

## Launch

```bash
bash train_sft.sh
```

Override dataset or model at runtime if needed:

```bash
TRAIN_DATA=/path/to/train.parquet \
VAL_DATA=/path/to/val.parquet \
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
NPROC_PER_NODE=8 \
bash train_sft.sh
```

## Teacher rollout

To distill from a deployed Search-R1 teacher served by vLLM, first generate raw trajectories:

```bash
.venv/bin/python scripts/sft/rollout_teacher_vllm.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --retriever-url http://127.0.0.1:8000/retrieve \
  --data-sources nq,hotpotqa \
  --samples-per-source 5000 \
  --concurrency 32 \
  --output-dir data/teacher_rollout
```

This writes:

- `raw_rollouts.jsonl`: every attempted trajectory
- `accepted_rollouts.jsonl`: trajectories that pass the built-in quality filter
- `summary.json`: aggregate counts

Each record stores the original prompt, full response trajectory, per-turn search logs, quality metrics, and stop reason. Concurrency is per-trajectory: each sample runs sequential search turns, while multiple samples roll out in parallel.

For a one-command pipeline from teacher rollout to parquet:

```bash
bash scripts/sft/collect_cold_start_data.sh
```

Common overrides:

```bash
TEACHER_BASE_URL=http://127.0.0.1:8001/v1 \
TEACHER_MODEL=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2 \
RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
DATA_SOURCES=nq,hotpotqa \
SAMPLES_PER_SOURCE=5000 \
CONCURRENCY=32 \
ROLLOUT_DIR=data/teacher_rollout \
PARQUET_DIR=data/search_sft \
bash scripts/sft/collect_cold_start_data.sh
```

## Raw to parquet

Convert the filtered rollout records into SFT parquet files:

```bash
.venv/bin/python scripts/sft/raw_to_parquet.py \
  --input-jsonl data/teacher_rollout/accepted_rollouts.jsonl \
  --output-dir data/search_sft \
  --require-quality-pass
```

The conversion step deduplicates by question, keeps the highest-scoring trajectory, and writes `train.parquet` plus `val.parquet`.
