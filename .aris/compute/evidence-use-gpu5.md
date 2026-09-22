# Existing .venv on physical GPU 5

Working directory: `/data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD`.
Spec: `.aris/compute/evidence-use-gpu5-spec.json`. Reuse installed packages without changes.
Inference: local 0.5B student checkpoints, BF16 SDPA, one GPU, two CPU threads.
GPU UUID: `GPU-2abc6681-2116-879c-a361-749edb32ac3b` (physical index 5).
Validation status: pending fresh-agent execution of the command below.

```bash
CUDA_VISIBLE_DEVICES=GPU-2abc6681-2116-879c-a361-749edb32ac3b HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/experiments/evidence_use/witness.py
```

Expected: exit 0 and `WITNESS_GPU5` with finite logits, nonempty generation,
and one A100 GPU visible. No training, package installation, or retriever restart.
The existing retriever at localhost:8000 is accessed through a requests Session
with trust_env=False to bypass outbound HTTP proxy settings for loopback.
