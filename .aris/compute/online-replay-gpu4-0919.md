# GPU4 continuation, Sept19

Cwd: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
Spec .aris/compute/online-replay-gpu4-0919-spec.json, canonical hash d2455624. Reuse existing conda searchr1 stack and isolated source; no installs. GPU4 only; preserve other processes. User explicitly removed remaining-free-memory guard. This wrapper logs GPU memory but NEVER gates startup or kills training based on memory thresholds (owned or free). STOP, actual training errors and completion cleanup still apply only to tagged own processes.

Run exactly once:
```bash
.venv/bin/python research_runs/online_replay_20260919/launch/run.py --mode witness --arm baseline --run-name witness_baseline
```
Launch via detached Python subprocess.Popen(stdin=DEVNULL,stdout=log,stderr=STDOUT,start_new_session=True,cwd=above). Log: reports/online_replay_gpu4_20260919/witness_baseline_supervisor.log. Do not use shell nohup background. Send parent PID/status immediately then monitor3actual updates and finite nonzero saved checkpoint delta. Save ENV_WITNESS_GPU4.md under this new report directory; PENDING until checks actually complete. Parent owns formal dispatch; do not manually launch baseline. No edits/installs/retries or killing other processes.

PriorGPU3 completed3updates and savedstep3 but postcheck failed because launch/queue.py shadowed stdlib queue during torch import. New directory has dispatch.py, no queue.py. Postcheck supervisor records verifying_checkpoint, and dispatch tracks supervisor PID rather than mistaking normally-finished training child for dead witness. Short hashed Ray temp path retained. Fullbatch16questions x8rollouts,3updates,train-only4casevalidation; formal50updates use frozen128development.
