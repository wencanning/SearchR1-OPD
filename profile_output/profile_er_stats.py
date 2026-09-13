"""Microbenchmark ER full-vocabulary target statistics on one GPU."""

import json
import time

import torch

from verl.utils.evidence_residual import compute_evidence_residual_target_stats


def main():
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    rows = 512
    vocab_size = 152064
    observed = torch.randn(rows, vocab_size, device=device, dtype=torch.float32) * 5.0
    hidden = torch.randn(rows, vocab_size, device=device, dtype=torch.float32) * 5.0
    labels = torch.randint(vocab_size, (rows,), device=device)

    # Initialize CUDA libraries before recording timings.
    compute_evidence_residual_target_stats(
        observed[:8], hidden[:8], labels[:8], token_chunk_size=8
    )
    torch.cuda.synchronize()

    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "rows": rows,
        "vocab_size": vocab_size,
    }))
    for chunk_size in (16, 32, 64, 128, 256, 512):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        start = time.perf_counter()
        stats = compute_evidence_residual_target_stats(
            observed,
            hidden,
            labels,
            token_chunk_size=chunk_size,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        checksum = sum(float(value.float().mean()) for value in stats.values())
        print(json.dumps({
            "chunk_size": chunk_size,
            "elapsed_s": elapsed,
            "rows_per_s": rows / elapsed,
            "incremental_peak_gib": (
                torch.cuda.max_memory_allocated() - baseline_memory
            ) / 1024**3,
            "checksum": checksum,
        }))


if __name__ == "__main__":
    main()
