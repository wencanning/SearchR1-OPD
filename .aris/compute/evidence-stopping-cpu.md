# CPU stopping diagnostics

Cwd: `/data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD`.
Reuse existing .venv without installs. Specification:
`.aris/compute/evidence-stopping-cpu-spec.json`.
No GPUs, no retrieval-service calls. Four CPU threads, nice10 for experiment worker.
Students: FP32 eager, standard HF generation cache. Teacher: FP32 eager, no cache,
already witnessed in reports/er_correction_cpu_20260914/ENV_WITNESS.md.
GPU2/3 sharing was conditionally allowed by user, but is not used by this plan.

Fresh-agent documented witness:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/experiments/evidence_cpu_followup/witness.py
```

Expected exit0 and CPU_STOPPING_WITNESS, finite FP32 logits, nonempty generation,
no CUDA initialized, cached/noncached first-pass logits agree within 1e-5.
