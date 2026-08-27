"""Profile exact-FP32 and TF32 ER teacher forwards on one A100."""

import argparse
import gc
import json
import time
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM

from verl.utils.evidence_residual import (
    build_evidence_hidden_attention_mask,
    compute_evidence_residual_target_stats,
)


MODEL_PATH = "/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3"
VOCAB_LIMIT = 151665


def synchronize_timed(callable_):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def make_batch(batch_size, sequence_length, response_length):
    input_ids = torch.randint(
        VOCAB_LIMIT,
        (batch_size, sequence_length),
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(sequence_length, device="cuda").expand(batch_size, -1)
    evidence_mask = torch.zeros_like(input_ids)
    evidence_mask[:, sequence_length // 3: sequence_length // 2] = 1
    hidden_mask = build_evidence_hidden_attention_mask(
        attention_mask,
        evidence_mask,
        torch.float32,
    )
    labels = torch.randint(
        VOCAB_LIMIT,
        (batch_size, response_length),
        device="cuda",
    )
    return input_ids, attention_mask, position_ids, hidden_mask, labels


def forward(model, batch, attention_mask, response_length, forward_dtype):
    input_ids, _, position_ids, _, _ = batch
    context = (
        nullcontext()
        if forward_dtype == "fp32"
        else torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if forward_dtype == "bfloat16" else torch.float16,
        )
    )
    with context:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            num_logits_to_keep=response_length + 1,
        ).logits[:, :-1]


def target_stats(observed, hidden, labels):
    return compute_evidence_residual_target_stats(
        observed,
        hidden,
        labels,
        token_chunk_size=512,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--response-length", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--precisions", choices=("highest", "high"), nargs="+", default=("highest", "high"))
    parser.add_argument(
        "--forward-dtypes",
        choices=("fp32", "bfloat16", "float16"),
        nargs="+",
        default=("fp32",),
    )
    args = parser.parse_args()
    print(json.dumps({
        "event": "environment",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "sequence_length": args.sequence_length,
        "response_length": args.response_length,
    }), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval().cuda()
    batches = {
        batch_size: make_batch(batch_size, args.sequence_length, args.response_length)
        for batch_size in args.batch_sizes
    }

    baseline_stats = {}
    for forward_dtype in args.forward_dtypes:
        for precision in args.precisions:
            torch.set_float32_matmul_precision(precision)
            for batch_size in args.batch_sizes:
                batch = batches[batch_size]
                _, attention_mask, _, hidden_mask, labels = batch
                # Warm each attention-mask path once at the smallest batch.
                if batch_size == args.batch_sizes[0]:
                    with torch.no_grad():
                        forward(model, batch, attention_mask, args.response_length, forward_dtype)
                        forward(model, batch, hidden_mask, args.response_length, forward_dtype)
                    torch.cuda.synchronize()

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    observed, visible_s = synchronize_timed(
                        lambda: forward(
                            model, batch, attention_mask, args.response_length, forward_dtype
                        )
                    )
                    hidden, hidden_s = synchronize_timed(
                        lambda: forward(
                            model, batch, hidden_mask, args.response_length, forward_dtype
                        )
                    )
                    stats, stats_s = synchronize_timed(
                        lambda: target_stats(observed, hidden, labels)
                    )

                record = {
                    "event": "measurement",
                    "precision": precision,
                    "forward_dtype": forward_dtype,
                    "output_dtype": str(observed.dtype),
                    "batch_size": batch_size,
                    "visible_s": visible_s,
                    "hidden_s": hidden_s,
                    "stats_s": stats_s,
                    "sequences_per_s": batch_size / (visible_s + hidden_s + stats_s),
                    "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }
                comparable = {
                    key: value.detach().cpu()
                    for key, value in stats.items()
                    if key in ("target_log_prob", "target_entropy")
                }
                if forward_dtype == "fp32" and precision == args.precisions[0]:
                    baseline_stats[batch_size] = comparable
                else:
                    for key, value in comparable.items():
                        delta = (value - baseline_stats[batch_size][key]).abs()
                        record[f"{key}_mae_vs_fp32"] = float(delta.mean())
                        record[f"{key}_max_vs_fp32"] = float(delta.max())
                print(json.dumps(record), flush=True)
                del batch, attention_mask, hidden_mask, labels, observed, hidden, stats
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
