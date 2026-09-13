# Local Compute Environments

### env: searchr1@3c1f7096

- how: conda environment `searchr1` in tmux pane `%1` on `inspur`
- tier: 2 × NVIDIA A100-SXM4-80GB, physical GPUs 2 and 3
- weights: local student `data/student/1B`; local 7B teacher checkpoint
- validated: 2026-08-24 (imports + seeded two-GPU matmul + clean fresh-agent witness + retriever HTTP 200)
- witness_log: `experiment_logs/er_opd_gpu23_preflight_20260824_1232.log`
- gotcha: retriever intentionally occupies about 18 GiB on each target GPU; ER sanity must account for shared-memory headroom
- contention: recheck GPU processes immediately before launch; unrelated jobs may start on physical GPUs 2/3 after preflight
