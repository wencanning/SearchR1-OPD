# GPU4 low-memory student collection witness

Cwd: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD. Spec recovery-collect-gpu4-spec.json hash 37ad3ae6. Existing .venv, no installs. PhysicalGPU4 only, student0.5B only. Allocator8GiB, ownprocess10GiB stop, devicefree16GiB floor, startup24GiB. Do not change other processes.

Run this invocation verbatim once:

```bash
.venv/bin/python scripts/experiments/recovery_collect_gpu4_20260917/guard.py scripts/experiments/recovery_collect_gpu4_20260917/probe.py --model student --out reports/recovery_collect_gpu4_20260917/witness_student --witness
```

Expect exit0, GPU4_RECOVERY_WITNESS_PASS student, complete1/1 with nonempty generation. One old diagnostic input is reused only for kernel witness, not new evidence. Save reports/recovery_collect_gpu4_20260917/ENV_WITNESS.md with actual results/memory; no code changes, package installs or retries.
