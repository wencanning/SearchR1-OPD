#DualGPU201-update training environment

Spec hash 29974161. Existing searchr1 conda; no installs. New dualGPU resource shape requires fresh agent-follows-doc witness.

Cwd: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
User authorized stopping only ER-OPD in tmux0 pane%5 after scripts ready. Parent handles stopping ER-OPD. Only after parent sends HANDOFF_READY_V3, the fresh agent must execute the documented invocation verbatim in pane%5 using tmux send-keys. Do not send Ctrl-C, retry, change source, or stop any process. Validate the first3-update run independently.

Documented invocation to execute verbatim in pane%5 after HANDOFF_READY_V3:
```bash
bash /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD/research_runs/objective_ablation_gpu23_20260920/launch_v3/start.sh
```

Pipeline first command:
```bash
/data/home/wencanning/miniconda3/envs/searchr1/bin/python /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD/research_runs/objective_ablation_gpu23_20260920/launch_v3/run.py --mode witness --arm joint --seed 20260917 --run-name witness_joint_v3
```
It then waits for ENV_WITNESS_PASS.json before exec dispatch.py. Fresh agent must check reports/objective_ablation_gpu23_20260920/witness_joint_v3/status.json,metrics.jsonl,checkpoint_delta.json and launch_manifest.json. Expect3actualorderedupdates,finiteactor/pg_loss and grad_norm,actualLR1e-6,zero ablation/advantage_identity_error,coef1/1,nonzero finite savedparameterdelta. Inspect nvidia-smi/supervisor GPU logs and verify both physicalGPUs2,3participated. RetrieverPID110019 must remain running. No free-memory protection line; telemetry sums2GPUmemory.

Write ENV_WITNESS_V3.md. If all pass, write ENV_WITNESS_PASS.json with state=pass,launcher_sha256,verifiedchecks,measuredmemory,limitations. The pipeline automatically launches201-updateformaltraining on release. On failure write a report and tell parent; do not retry or release.

State pending.

V1 failed before any update because microbatch1 was divided byDP2to0. V2 sets actor/rollout/ref globalmicrobatch2, preserving per-rank1. Old witness_joint preserved. No source mathematics changed. Parent has already completed handoff; pane%5 currently shell. After HANDOFF_READY_V3, execute start.sh once and verify v3 outputs.

V3resource-tuning userrequest: actor,rollout-logprob,teacher globalmicro8/perrank4; global128rollouts and one optimizerupdate unchanged. V2three-step check runs tocompletion with no commonrelease; parent stops only its post-witness waiting shell before v3handoff. Do not launch before HANDOFF_READY_V3.
