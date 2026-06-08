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
