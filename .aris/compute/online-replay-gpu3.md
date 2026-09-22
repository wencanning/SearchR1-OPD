# GPU3 full-online witness and serial experiment queue

Cwd: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
Spec: .aris/compute/online-replay-gpu3-spec.json; canonical hash efdcc75d. Existing searchr1 conda runtime; no installs. Teacher BF16 resident, same isolated source as GPU4. GPU3 exclusively for new training runs; existing retrieval PID110019 and unrelated processes preserved. Startup54GiB free, training own cap48GiB, minimum card free10GiB. Polling guards are not hard reservations.

Run exactly once:

```bash
.venv/bin/python research_runs/online_replay_20260918/launch/run.py --mode witness --arm baseline --run-name witness_baseline
```

Launch durably via Python subprocess.Popen with cwd as above, stdin=DEVNULL, stdout opened to reports/online_replay_gpu3_20260918/witness_baseline_supervisor.log, stderr=STDOUT, start_new_session=True. Do not use shell nohup-and-ampersand: the prior launch shell died before Python started. Report PID and running status promptly; monitor this witness. Do not launch formal runs yourself: parent serial queue owns them.

Expected: three updates at16questions x8rollouts/mini128, actual LR1e-6, checkpoint step3 with finite nonzero tensor delta, sentinel FULL_ONLINE_GPU3_WITNESS_PASS, baseline_WITNESS_PASS.json with state complete updates3. Four TRAIN-only validation cases for witness; formal runs use128development cases. Save ENV_WITNESS_GPU3.md under reports/online_replay_gpu3_20260918 with exact evidence. Before completion report PENDING, not PASS. On failure report full cause without retries/source edits/installs or killing other jobs.

Previous GPU4 failures: 40GiB owned-memory guard at second-step reference reload; mitigated by resident teacher and larger available resource window. Later resident-teacher witness failed before GPU allocation due AF_UNIX107-byte Ray path limit. This launcher uses a short hashed /tmp/r3-XXXXXXXX path. Neither old failure is counted as scientific effectiveness evidence. Baseline/method runs restart originalstep50 weights with fresh AdamW, not restore failed checkpoint.
