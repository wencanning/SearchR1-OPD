# GPU4 recovery: full-batch multistep witness

Cwd: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
Spec: .aris/compute/online-replay-gpu4-spec.json; canonical hash 35c852e7. Same existing conda stack, no installs. Teacher reference now stays resident on GPU4 (BF16, no manual offload). Only GPU4 is permitted; GPU3 disabled; shared retrieval untouched. Student collector (~2GiB) continues. Teacher collection must finish before training. Startup requires64GiB free, own training process cap60GiB, minimum card free16GiB. Limits are polling guards, not allocator reservations.

Run exactly once, in a durable background process with output to reports/online_replay_gpu4_20260917/recovery_pipeline.log:

```bash
.venv/bin/python research_runs/online_replay_20260917/launch/recover_pipeline.py
```

This invocation waits for resumed teacher completion512, then free-memory preflight, runs witness_v4_resident_teacher with16questions x8rollouts for THREE real optimizer updates, verifies saved step3 parameters changed and finite metrics, then starts the authorized50-update baseline_seed20260917_restart1 from original step50 weights with fresh optimizer. Witness has4train-only validation cases; formal baseline uses frozen128development cases. All training sizes/settings match. Both use seed20260917. Expected witness sentinel FULL_ONLINE_GPU4_WITNESS_PASS; new MULTISTEP_WITNESS_PASS.json (old single-step marker is not accepted).

Report launch process and observed queue state promptly to parent, then monitor witness status. A queued/running witness is PENDING, not PASS. On witness completion verify3 updates, checkpoint delta and guard stats. Save fresh-agent report to reports/online_replay_gpu4_20260917/ENV_WITNESS_V4.md. Do not rerun, fix source, install packages, kill jobs, or manually launch baseline; durable pipeline owns baseline. Pipeline stops on any failure; it never retries automatically. A root reports/online_replay_gpu4_20260917/STOP prevents launch or asks current guarded run to stop.

Incident: old single-step witness missed step2 peak51,102MiB, above own40GiB guard. Concurrent teacher free-memory guard also fired. Manual FSDP offload copies param.data and _local_shard separately: plausible duplicate storage/reload contributor, not yet measured causal attribution. Resident teacher removes this path; serial scheduling removes collector collision. New witness tests actual repeated steps; no claim of validated recovery until it passes. Old baseline has1 update and no saved checkpoint; restart is not an exact resume.
