# CPU diagnostic environment

### env: er-cpu@365102af

Existing environment reused; seeded CPU witness and fresh-agent invocation passed
2026-09-14. Witness report: `reports/er_correction_cpu_20260914/ENV_WITNESS.md`.
No package, model-weight, or training-configuration changes.

Reuse the existing repository `.venv`; no packages are installed or upgraded.
Working directory: `/data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD`.
Specification: `.aris/compute/er-cpu-env-spec.json`.
Teacher is a local frozen Qwen2.5 7B checkpoint. Real experiments use CPU FP32
eager attention, no cache, at most 8 PyTorch threads and nice level 10.
CUDA_VISIBLE_DEVICES must be empty. No training process or retriever is changed.

Documented seeded witness invocation:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/diagnostics/witness_er_cpu.py
```

Expected: one `WITNESS_CPU` line, device cpu, null mask error <1e-6,
nonzero hidden-evidence effect, CUDA uninitialized. This validates the tiny
model execution path; full teacher parity with saved log probabilities is
a separate per-case gate before interpreting scores.
