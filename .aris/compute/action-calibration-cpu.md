# CPU action calibration and local-update environment

- Environment: action-calibration-cpu@ab843930
- Existing repository .venv; no installs, CPU FP32 eager, 4 threads, CUDA hidden.
- Disposable 0.5B forward/backward/AdamW witness; no on-disk weight update, retrieval request, or GPU calls.
- Spec: .aris/compute/action-calibration-cpu-spec.json
- Validation pending fresh-agent invocation below.

Run verbatim from /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/experiments/action_calibration_cpu/witness.py
```

Expected sentinel: CPU_BACKWARD_WITNESS_PASS. Output: reports/action_calibration_cpu_20260916/ENV_KERNEL_WITNESS.json, passed=true and cuda_initialized=false.
Report divergence and stop if invocation fails; do not install or change packages.
