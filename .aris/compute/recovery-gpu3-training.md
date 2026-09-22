# GPU3 local training witness

Working directory: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
Existing .venv; no rebuild/packages. Extended compute spec recovery-gpu3-training-spec.json, canonical hash 5d4b3d9b. Same24GiB allocator/26GiB own process/16GiB minimum cardfree. ONE owned worker at a time: run ONLY after recovery_suite launcher and all its owned worker descendants have exited. Do not kill other processes. No retriever calls for this witness.

Run verbatim:

```bash
.venv/bin/python scripts/experiments/recovery_20260917/guard.py scripts/experiments/recovery_20260917/local_update.py --witness --out reports/recovery_20260917/witness_training
```

Expected: exit0, TRAIN_WITNESS, finite loss/gradient, nonzero parameter delta, status complete, actual full student forward/backward/Adam update. Witness saves no weights and modifies no existing checkpoint. Inspect memory watchdog and save TRAIN_ENV_WITNESS.md in reports/recovery_20260917. If any divergence report it; no installs or code fixes. This witness proves dispatch and optimizer path, not4K-context memory feasibility nor method effectiveness.
