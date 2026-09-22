# Dual GPU sampling-corrected 201-update experiment
Spec hash 9a411abf. Existing searchr1 Python environment, no installs.
Cwd /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD.
Physical GPUs2/3 authorized; preserve shared retrieverPID110019 and all unrelated jobs. No free-memory cutoff. Per-rank actor/ref/logprob microbatch4 (global8), global128trajectories=16questions x8rollouts, one optimizer update. Formalrun201updates from existingstep50weights with freshAdamW.

After parent sends HANDOFF_READY_V4, fresh agent executes this exact invocation once in tmux pane%5:
```bash
bash /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD/research_runs/objective_ablation_gpu23_20260920/launch_v4/start.sh
```
No Ctrl-C/retries/edits/kills by witness agent. Verify reports/objective_ablation_gpu23_20260920/witness_joint_v4: status complete, exactly3ordered optimizerupdates with128rollouts, finite loss/grad, actualLR1e-6, coefficients1/1 and identity0, finite nonzero checkpointdelta, both physicalGPU allocations, retrieveralive, source/launcherhashes unchanged. Verify16trace records align across masks and advantages.
Additional scientific acceptance: runtime SamplingParams seed must be absent orNone while engine seed remains fixed; sampling/unique_sequences_per_group_mean must exceed1 in eachwitness update, selectedgroups should show distinct sequences. Report actual diversity and rewardzero metrics, do not require allgroups mixed outcomes.
Write ENV_WITNESS_V4.md. If all pass, write common ENV_WITNESS_PASS.json with state=pass, launcher_sha256 and checks and measured memory; pipeline then starts formal201queue automatically. Onfailure do not release or retry; reportparent.
V1 failed zero microbatch; V2 functionalpassed; V3 resource witness uses duplicated-requestseed and cannot release formalqueue. V4 is isolated source with minimal seed-copy exclusion plus per-update diversity telemetry. Previous results retain provenance and are not pooled with sampling-corrected runs.
