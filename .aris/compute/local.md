# Local Compute Environments

### env: searchr1@3c1f7096

- how: conda environment `searchr1` in tmux pane `%1` on `inspur`
- tier: 2 × NVIDIA A100-SXM4-80GB, physical GPUs 2 and 3
- weights: local student `data/student/1B`; local 7B teacher checkpoint
- validated: 2026-08-24 (imports + seeded two-GPU matmul + clean fresh-agent witness + retriever HTTP 200)
- witness_log: `experiment_logs/er_opd_gpu23_preflight_20260824_1232.log`
- gotcha: retriever intentionally occupies about 18 GiB on each target GPU; ER sanity must account for shared-memory headroom
- contention: recheck GPU processes immediately before launch; unrelated jobs may start on physical GPUs 2/3 after preflight

### env: evidence-use-gpu5@87e08aea

- how: existing repository .venv, Python 3.12.13, no installs
- tier: 2 CPU threads, physical GPU 5 only (A100 80GB), BF16 SDPA
- validated: 2026-09-15; fresh-agent real checkpoint forward and generation passed
- witness: reports/evidence_use_gpu5_20260915/ENV_WITNESS.md
- retriever: localhost:8000 HTTP 200 verified separately with requests trust_env=False
- spec: .aris/compute/evidence-use-gpu5-spec.json

### env: evidence-stopping-cpu@a33f1b1d

- how: existing .venv, no package changes
- tier: 4 CPU threads, nice10, no GPUs, sequential student/teacher inference
- validated: 2026-09-15; fresh-agent witness passed, CPU FP32, CUDA uninitialized
- witness: reports/evidence_cpu_followup_20260915/ENV_WITNESS.md
- spec: .aris/compute/evidence-stopping-cpu-spec.json

### env: action-calibration-cpu@ab843930

- how: existing .venv, no package changes
- tier: 4 CPU threads, nice10, 65 GiB estimated peak, no GPUs
- validated: 2026-09-16; fresh-agent disposable 0.5B FP32 eager forward/backward/AdamW witness passed; CUDA uninitialized
- witness: reports/action_calibration_cpu_20260916/ENV_WITNESS.md
- spec: .aris/compute/action-calibration-cpu-spec.json

### env: fork-cpu-retriever@0e8af9ff

- Existing retriever conda env, Python3.10, torch2.4.0, faiss1.8.0, transformers4.57.6, datasets4.8.5; no installs.
- Validated 2026-09-16 by fresh-agent CPU_RETRIEVER_WITNESS_PASS, exit0; real61GiB index, 21,015,324 corpus records, real E5 query; CUDA false.
- Witness: reports/er_next_20260916/cpu_retriever_witness/ENV_WITNESS.md. Initial cold setup659s; CPUquery24s.
- CPU4 threads, corpus loader automatically single-process for single JSON shard. FP32 CPU retrieval differs from historical FP16 GPU index.

### env: fork-student-gpu5@9b8820da

- Existing .venv, BF16 eager,4threads; physicalGPU5 only.
- Validated 2026-09-16 fresh-agent real0.5B forward/generate; peakallocated1.021GiB, CUDAtrue.
- Witness: reports/er_next_20260916/gpu_fork_witness/ENV_WITNESS.md.
- Allocator6GiB, ownprocess8GiB stop, free12GiB stop; never stop other processes.

### env: recovery-gpu3@f4f26047

- Existing .venv BF16 SDPA, GPU3 only, no installs.
- Validated real seeded student and teacher generations, exit0, GPU3_RECOVERY_WITNESS_PASS. Teacher allocated peak15,428,727,296bytes; observed process15,390MiB.
- Independent witness executed verbatim; reused prior witness-agent because fresh spawn hit thread limit, so not a fresh-context certification.
- Witness: reports/recovery_20260917/ENV_WITNESS.md. Allocator24GiB, own26GiB and free16GiB stop, startupfree32GiB.

### env: recovery-gpu3-training@5d4b3d9b (2026-09-17)
Existing .venv; extended inference spec with FP32 student parameters and AdamW, BF16 autocast, gradient checkpointing and chunked head. No package/environment rebuild. Documented invocation executed verbatim by non-author reused gpu_fork_witness after inference suite exited. TRAIN_WITNESS exit0; finite loss/gradient; max parameter change1.01328e-6; BF16 changed fraction0.0327289. Peak CUDA9,028,414,976B; process9,514MiB; cardfree minimum54,206MiB. Report reports/recovery_20260917/TRAIN_ENV_WITNESS.md. Reused agent due earlier thread capacity; not a fresh-context validation. Short-context witness does not certify arbitrary4K training peak. Actual jobs remain guarded24GiB allocator/26GiB own-process/16GiB min cardfree. Original weights read-only.

### env: recovery-gpu4@1a7d7cb6

- Existing repository .venv, no installs; physicalGPU4, BF16 SDPA,4CPUthreads.
- Guard: allocator20GiB/own22GiB/minfree16GiB/startfree34GiB.
- Fresh agent-follows-doc teacher forward/generation passed; reports/recovery_teacher_start_gpu4_20260917/ENV_WITNESS.md. Peak allocated14.37GiB, process15390MiB, minfree22543MiB.
- Question-start TRAIN-only demonstrations launched; no student training. ExistingGPU2/3retrieval reused.

### GPU4 contention and student fallback,2026-09-17

- Teacher full collection stopped by owned-process guard when other job growth left15009MiB free; no other processes terminated. Witness remains valid but resource availability was not stable.
- Low-memory student spec37ad3ae6: allocator8GiB/own10GiB/minfree16GiB/startfree24GiB. Fresh witness passed,1040597504B allocated,1556MiB process peak.
- GPU4 now collects the existing queue first states; complete-only provenance import automatically resumes frozenGPU3 waiter. No software installation or full training.

### env: online-replay-gpu4@63adbd49 (2026-09-17)

- Existing conda searchr1, torch2.4.0+cu121/transformers4.47.1/ray2.51.2/flash_attn2.8.3; installed patched vLLM. No installs. GPU4 numeric CVD4 checked against UUID.
- Fresh agent-follows-doc V3 full online rollout/retrieval/teacher target/actor optimizer/save/eval passed. Actual pre-updateLR1e-6; finite nonzero changes in291/291 saved tensors. Peak owned32196MiB; minimum cardfree30383MiB. Witness32rollouts; earlierV2 exercised128rollouts peak owned32210MiB but had firstLR0 and is NOT a parameter-update pass.
- Isolated zero-warmup scheduler fix and actualLR log; root source unchanged. Guard maxowned40GiB/minfree20GiB/startfree48GiB. Report reports/online_replay_gpu4_20260917/ENV_WITNESS_V3.md.
- Formal baseline50updates started19:15; B/C not started. This witness does not certify positive auxiliary CE or multi-GPU communication.
