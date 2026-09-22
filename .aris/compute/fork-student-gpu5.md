# GPU5 fork student environment: witness pending

Working directory: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD.
Spec .aris/compute/fork-student-gpu5-spec.json. Existing .venv, no installs.
User authorizes using available GPU5 memory, never disturbing others. Our wrapper binds UUID, sets student allocator6GiB, stops only its own process group at process8GiB or cardfree12GiB; requires free20GiB before launch. Current other process around30GiB on80GiB card. Sampling watchdog has delay; no claim of absolute scheduling isolation.
Run verbatim:

```bash
.venv/bin/python scripts/experiments/er_next/run_gpu5_guarded.py --states reports/er_next_20260916/fork_states.frozen.json --checkpoint verl_checkpoints/opd-grpo-0.5B/actor/global_step_50 --out reports/er_next_20260916/gpu_fork_witness --limit 1 --replicates 1 --witness-only
```

Expected exit0, GPU_FORK_WITNESS_PASS and WITNESS.json with CUDA true, finite896-wide hidden state, nonempty generation. No teacher, retrieval, training or checkpoint writes. Do not modify code or improvise; report divergence.
