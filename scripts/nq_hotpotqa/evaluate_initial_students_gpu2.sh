#!/usr/bin/env bash
# Sequential evaluation of stock Qwen2.5-Instruct 0.5B/1.5B on physical GPU 2.
# No project SFT/RL checkpoints. These are upstream instruction-tuned models,
# not the pretrained-only Qwen2.5 Base variants.
# No training updates; uses evaluate.sh's val_only=true mode.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES=2
export N_GPUS=1
export PYTHON_BIN=${PYTHON_BIN:-/data/home/wencanning/miniconda3/envs/searchr1/bin/python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DATA_DIR="$PWD/data/nq_hotpotqa_train"
export VAL_FILE="$DATA_DIR/test.parquet"
export EVAL_SEED=42
export EVAL_DO_SAMPLE=true
export RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
for model in "/data/home/wencanning/models/Qwen2.5-0.5B-Ins" "/data/home/wencanning/models/Qwen2.5-1.5B-Ins"; do
    for file in "$model/config.json" "$model/tokenizer_config.json"; do
        [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
    done
    compgen -G "$model/*.safetensors" >/dev/null || { echo "Missing weights: $model" >&2; exit 1; }
done
for file in "$VAL_FILE" "$DATA_DIR/train.parquet"; do
    [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done

RESULT_DIR=${RESULT_DIR:-"$PWD/eval_loop_logs/vanilla_instruct_students_gpu2_$(date +%Y%m%d_%H%M%S)_$$"}
mkdir -p "$RESULT_DIR"
echo "Results: $RESULT_DIR"
for size in 0.5B 1.5B; do
    if [[ "$size" == 0.5B ]]; then
        export BASE_MODEL="/data/home/wencanning/models/Qwen2.5-0.5B-Ins"
    else
        export BASE_MODEL="/data/home/wencanning/models/Qwen2.5-1.5B-Ins"
    fi
    model_out="$RESULT_DIR/$size"
    mkdir -p "$model_out"
    echo "Evaluating stock Qwen2.5-Instruct $size: $BASE_MODEL on GPU $CUDA_VISIBLE_DEVICES"
    bash scripts/nq_hotpotqa/evaluate.sh \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
        2>&1 | tee "$model_out/eval.log"
    "$PYTHON_BIN" - "$model_out/eval.log" "$model_out/completed.json" "$BASE_MODEL" <<'PY'
import json
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(errors='replace')
text = re.sub(r'\x1b\[[0-9;]*m', '', text)
text = re.sub(r'\(main_task pid=\d+\)\s*', '', text)
text = text.replace("'", '').replace('"', '')
pairs = re.findall(r'val/test_score/([\w/-]+)\s*:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)', text)
metrics = {key: float(value) for key, value in pairs}
required = {'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'Avg'}
if not required.issubset(metrics):
    raise RuntimeError(f'Incomplete evaluation results; missing: {sorted(required - metrics.keys())}')
result = {'model': sys.argv[3], 'metrics': metrics}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
done
echo "Both evaluations completed: $RESULT_DIR"
