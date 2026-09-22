# EROPD 0.5B alpha sweep

Completed 2026-09-22 01:18 (Asia/Shanghai). All five runs exited 0 and completed 102/102 validation batches (51,713 examples). Seed 42, sampling enabled, temperature 1, validation batch 512, two GPUs, tensor parallel size 1, rollout memory utilization 0.5. Scores below and in summary.csv are fractions, not percentages.

| Alpha directory | Step | nq | triviaqa | popqa | hotpotqa | 2wikimultihopqa | musique | bamboogle | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 150 | 0.339335 | 0.510121 | 0.408004 | 0.309656 | 0.289679 | 0.095573 | 0.232000 | 0.312053 |
| 0.5 | 150 | 0.333795 | 0.506055 | 0.402818 | 0.315192 | 0.280614 | 0.089367 | 0.232000 | 0.308549 |
| 1 | 150 | 0.327424 | 0.495890 | 0.394126 | 0.308575 | 0.284987 | 0.096400 | 0.216000 | 0.303343 |
| 1.5 | 100 | 0.329363 | 0.499072 | 0.408004 | 0.295746 | 0.259622 | 0.078610 | 0.160000 | 0.290060 |
| 2 | 100 | 0.332410 | 0.485636 | 0.411089 | 0.292910 | 0.258190 | 0.074059 | 0.192000 | 0.292328 |

## Provenance and interpretation

- Alpha 0 is the observed OPD checkpoint. The project README intentionally uses a different display mapping (OPD row ← alpha0.5, ER-OPD row ← alpha0); consult these source records for actual checkpoint identity.
- Directory alpha2 has saved training configuration evidence_residual_alpha=1.5. Its alpha2 identity is unverified; do not treat it as a verified alpha2 ablation.
- The first three checkpoints are step150; the last two are step100. This is not a training-step-matched alpha ablation.
- These are single-seed evaluations. No significance or cross-seed stability claim is supported.
- Per-alpha JSON files retain repository-relative checkpoint/source paths and SHA-256 of original completed.json. Full logs and weights remain local. Baseline JSON files are archived in ../baselines/.

## Reproduce evaluation

Prepare the checkpoint and datasets at the paths in manifest.json, start the retrieval service at http://127.0.0.1:8000/retrieve, and activate the existing Search-R1 environment. Only run when the target GPUs are available. For each checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2
export BASE_MODEL=verl_checkpoints/opd-grpo-0.5B/actor/global_step_150
export EVAL_SEED=42 EVAL_DO_SAMPLE=true
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export SEARCHR1_RAY_CPUS=16
export RAY_TMPDIR="/tmp/eropd_eval_${USER}_$$"
bash scripts/nq_hotpotqa/evaluate.sh \
  data.val_batch_size=512 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

Use a short RAY_TMPDIR: long workspace paths exceeded the Unix socket path limit in the earlier failed attempt. The successful run used the same training/evaluation source but a short temporary directory. The archived manifest records the completed run rather than an active scheduling request.
