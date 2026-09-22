# GPU4 teacher demonstration inference

Cwd: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
Spec: .aris/compute/recovery-gpu4-spec.json; hash 1a7d7cb6. Existing .venv, no package changes. Physical GPU4 only; sharedGPU2/3retrieval preserved. Cap20GiB allocator/22GiB owned process; at least16GiB device free; startup34GiB free. Guard terminates its own worker only. Do not run any other GPU task.

Run this exact invocation once; do not modify code or environment:

```bash
.venv/bin/python scripts/experiments/recovery_teacher_start_gpu4_20260917/guard.py scripts/experiments/recovery_teacher_start_gpu4_20260917/probe.py --model teacher --out reports/recovery_teacher_start_gpu4_20260917/witness_teacher --witness
```

Expected exit0 and GPU4_RECOVERY_WITNESS_PASS, status complete, nonempty generated IDs. This reuses one old diagnostic input only as a kernel witness, not new scientific data. Teacher real seeded forward/generation; no training. If guard says insufficient free memory, report resource blocking; do not retry, kill others or install packages. Save result in reports/recovery_teacher_start_gpu4_20260917/ENV_WITNESS.md.
