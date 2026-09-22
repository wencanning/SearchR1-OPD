# GPU3 recovery diagnostic environment

Working directory: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
Existing repository .venv, no installs. Spec: recovery-gpu3-spec.json, hash f4f26047. PhysicalGPU3 only. Teacher and student run sequentially. Reuse existing GPU2/3 retrieval service only when requested; this witness does not call retrieval. Allocator24GiB, ownprocess26GiB stop, cardfree16GiB stop, startupfree32GiB. Stops own child only. Other jobs and existing retriever are preserved.

Run these TWO invocations sequentially verbatim, without modifications:

```bash
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/probe.py --model student --out reports/recovery_20260917/witness_student --witness
```

```bash
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/probe.py --model teacher --out reports/recovery_20260917/witness_teacher --witness
```

Expected each exit0 with GPU3_RECOVERY_WITNESS_PASS and status complete. Each does one actual checkpoint forward/generation with seed; nonempty text required, correctness is not an environment check. If first fails, stop; report exact divergence and do not fix code or install packages. Save ENV_WITNESS.md under reports/recovery_20260917 with both observations and measured CUDA memory. No training/checkpoint writes.
