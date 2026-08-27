# On-Policy Distillation

Search-R1 supports On-Policy Distillation (OPD) with an independent teacher
checkpoint. The student generates normal Search-R1 trajectories, including
search tool calls. The teacher scores the same sampled tokens, and the actor is
updated with the K1 sampled reverse-KL signal:

```text
advantage = log p_teacher(token | context) - log p_student(token | context)
```

This minimizes `KL(student || teacher)` on contexts visited by the student.
OPD does not use a critic. It can optionally combine the distillation signal
with the group-normalized Search-R1 outcome reward:

```text
combined_advantage =
    distillation_coef * beta * opd_advantage
    + grpo_reward_coef * grpo_outcome_advantage

beta = beta_min + (beta_max - beta_min) * sigmoid(-gamma * task_advantage)
```

## Requirements

- Use the FSDP actor strategy.
- Teacher and student checkpoints must use exactly the same tokenizer
  vocabulary.
- Set `actor_rollout_ref.ref.model_path` to the teacher checkpoint.
- Keep `actor_rollout_ref.actor.use_kl_loss=false`.
- Set `actor_rollout_ref.actor.entropy_coeff=0.0` for pure distillation, or use
  a nonzero value only when explicit entropy regularization is desired.

## Configuration

```yaml
algorithm:
  adv_estimator: opd
  opd:
    advantage_mode: token
    normalize: false
    clip_value: null
    distillation_coef: 1.0
    grpo_reward_coef: 1.0
    use_gated_distillation: true
    gamma: 1.0
    beta_min: 0.0
    beta_max: 0.05
    rce:
      enable: false
      entropy_normalization: percentile_rank
      w_min: 0.1
      w_max: 1.0
      alpha: 4.0
      tau: 0.0
      default_retrieval_hit: 0.5

actor_rollout_ref:
  actor:
    clip_ratio: 0.2
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    clip_ratio_c: 3.0
  ref:
    model_path: /path/to/teacher
```

`advantage_mode=token` distills each sampled token independently.
`advantage_mode=sequence` assigns each token the cumulative future K1 signal,
which also propagates teacher preference for later tokens to earlier actions.
Use `clip_value` to limit high-variance teacher/student log-probability gaps.

Set `grpo_reward_coef=0.0` for pure OPD. When it is nonzero, configure more
than one rollout per prompt, such as `actor_rollout_ref.rollout.n_agent=8`.
The Search-R1 rule-based final-answer reward is normalized within each prompt
group and added to the OPD advantage. With `use_gated_distillation=true`, the
OPD component is scaled by a per-sequence sigmoid beta gate from the normalized
task advantage. This mirrors the legacy gated OPD branch in the SOD code, but
the SOD `run_sod.sh` main path enables step-wise weighting instead. Observation
tokens are excluded from both advantage components and all related metrics.

`use_gated_distillation` is only active when `grpo_reward_coef != 0`. If
`grpo_reward_coef=0.0`, OPD is plain reverse-KL distillation regardless of the
gate setting:

```text
combined_advantage = distillation_coef * opd_advantage
```

## SOD baseline

This repository includes a Search-R1 adaptation of
[SOD: Step-wise On-policy Distillation for Small Language Model Agents](https://github.com/YoungZ365/SOD),
based on upstream commit `110c4b8`. The adaptation keeps Search-R1's existing
rollout, teacher scoring, GRPO, and actor-update paths. Only the released SOD
step-wise weighting rule is added.

Search-R1's `loss_mask` is one on assistant tokens and zero on retrieved
observations, so each contiguous run of ones is one SOD step. For every step:

```text
d_k = mean_token |log p_student - log p_teacher|
w_1 = 1
w_k = min(
    product_{u=1}^{k-1} (d_u + epsilon) / (d_{u+1} + epsilon),
    1 + delta,
)

combined_advantage =
    lambda_distill * w_k * opd_advantage
    + grpo_reward_coef * grpo_outcome_advantage
```

The official defaults are `epsilon=1e-6` and `delta=0.2`. A growing
student-teacher gap downweights later teacher supervision; when the policies
realign, the weight can recover up to `1 + delta`. Observation and padding
tokens have zero SOD weight. Protocol tags remain included, matching the
upstream definition over all assistant tokens.

SOD is isolated behind a default-off configuration:

```yaml
algorithm:
  adv_estimator: opd
  opd:
    teacher_target: observed
    advantage_mode: token
    normalize: false
    clip_value: null
    grpo_reward_coef: 1.0
    use_gated_distillation: false
    rce:
      enable: false
    sod:
      enable: true
      epsilon: 1.0e-6
      delta: 0.2
```

Run the dedicated baseline launcher with:

```bash
./train_sod.sh

# Optional path or hyperparameter overrides
STUDENT_MODEL=/path/to/student \
TEACHER_MODEL=/path/to/teacher \
SOD_DELTA=0.2 \
./train_sod.sh
```

Its defaults match the 1B two-GPU `train_er_opd.sh` setup wherever the SOD
method does not define a different variable. Existing OPD and ER-OPD launchers
keep `algorithm.opd.sod.enable=false` through the shared default and do not
enter this branch. Monitor `sod/stepwise_weight`, `sod/step_divergence`,
`sod/downweighted_token_fraction`, and `sod/upweighted_token_fraction`.

## Evidence-Residual OPD

Evidence-Residual OPD (ER-OPD) is an OPD teacher-target variant. It does not
use DGPO, gates, thresholds, or token-dependent weights. It can be run as pure
distillation or combined with vanilla GRPO outcome advantages. For every
student policy-token row, the frozen teacher is evaluated
twice on the same token ids and position ids:

```text
z_observed = teacher(full student trajectory)
z_hidden   = teacher(same trajectory, retrieved-content key columns blocked)
q_ER       = softmax(2 * z_observed - z_hidden)
```

The hidden pass uses a per-sample 4D additive causal mask. Text inside
`<information>...</information>` is hidden at every layer; the boundary tags,
sequence slots, causal order, padding, and position ids remain unchanged. The
two passes are independent and do not share a KV cache.

Strict ER/control runs load the teacher with FP32 FSDP parameters and produce
FP32 logits for both paired passes. Approximate target precision is rejected
unless `ref.allow_approximate_target_precision=true`. The optimized
`train_er_opd.sh` launcher explicitly opts into FP16 teacher parameters and
paired forwards; the evidence residual, full-vocabulary log-softmax, entropy,
and sampled-token extraction are still computed in FP32. This mode is labeled
separately because it is a measured speed/precision tradeoff, not bitwise
equivalent to the strict target. The reference worker returns only sampled-token
`log q_ER` plus audit metrics. This is exactly
the quantity needed by the existing on-policy K1 reverse-KL estimator and
avoids transferring an `[policy_tokens, vocab_size]` tensor through Ray. Target
normalization is chunked over token rows, never over the vocabulary.
Here `vocab_size` is the exact shared tokenizer coordinate set; unused padded
rows in differently sized Qwen output heads are excluded from both student and
teacher normalization.
Policy-token coverage follows executed Search-R1 trajectories: all retained
reasoning/search/answer tokens and a sampled final-answer EOS are included.
Suffixes sampled after an action-closing tag are rollout-engine over-generation
and are discarded before the action is executed.

ER-OPD enforces the following target invariants at startup:

- token-mode OPD with no whitening or clipping;
- one fixed positive `lambda_distill` and one PPO epoch;
- `rollout.temperature=1`, `state_masking=true`, and teacher SDPA attention;
- exact student/teacher id-to-token, added-token, and special-token alignment;
- one global mean over valid policy tokens across micro-batches and GPUs.

With `grpo_reward_coef>0`, the trainer requires more than one rollout per
question and combines the two signals as
`lambda_distill * ER_advantage + grpo_reward_coef * GRPO_advantage`. With
`grpo_reward_coef=0`, it instead requires exactly one rollout and one full-batch
optimizer update, preserving the stricter pure-distillation control.

The dedicated ER launcher filters the mixed training parquet to
`data_source=hotpotqa`. By default it validates on the original unfiltered
`data/nq_hotpotqa_train/test.parquet` benchmark used by Search-R1; set
`VAL_FILE` and/or `VAL_DATA_SOURCE` to override that behavior. The generic
`train_opd.sh` keeps its configured validation path unless overridden.

Run it with:

```bash
DATA_DIR=/path/to/hotpotqa \
STUDENT_MODEL=/path/to/Qwen2.5-1.5B-Instruct \
TEACHER_MODEL=/path/to/Qwen2.5-7B-Instruct \
./train_er_opd.sh
```

### Entropy-matched control

The matched control uses the same policy-token rows and fixed coefficient but
replaces the target with `softmax(z_observed / tau)`. `tau` must be calibrated
once on the declared frozen-student calibration split and then passed unchanged:

```bash
./calibrate_er_entropy.sh
ENTROPY_MATCHED_TAU=0.73 ./train_entropy_matched_opd.sh
```

Calibration writes both a timestamped audit record and the fixed
`refine-logs/ER_ENTROPY_CALIBRATION.json`; copy its printed temperature into
`ENTROPY_MATCHED_TAU` for the control run.

`verl.utils.evidence_residual` provides deterministic 5% calibration-id
selection and an automatically bracketed bounded Brent solver in log-temperature
space. The caller supplies a streaming mean-entropy function, so calibration
does not require retaining full-vocabulary logits for the entire split in RAM.

## Retrieval-Conditioned Entropy Gate

RCE-OPD adds a separate tensor-level interface,
`compute_rce_opd_advantage(...)`, for retrieval-conditioned entropy-gated OPD:

```text
rce_weight =
    w_min + (w_max - w_min)
    * sigmoid(alpha * (retrieval_hit - normalized_teacher_entropy - tau))

rce_opd_advantage =
    rce_weight * (log p_teacher(token | context) - log p_student(token | context))
```

If `step_ids` are provided, teacher entropy is averaged per step before
normalization and the same step weight is applied to all trainable tokens in
that step. Retrieved `<information>...</information>` tokens remain excluded by
`loss_mask`.

The current trainer path creates `rce_step_ids` by decoding Search-R1 tags.
Step 0 covers tokens before any completed `<information>` block; step 1 covers
tokens after the first completed information block; and so on. For each
information block, the trainer checks whether it contains a gold target string
and uses that block-level hit as `r_{k-1}` for the following step. This makes the
experiment runnable with the existing rollout data while preserving the
"previous retrieved evidence" semantics. A future rollout-side per-search hit
tensor can be passed to the same interface as `rce_retrieval_hit` without
changing the loss implementation.

For trainer-created RCE metadata, `default_retrieval_hit` is used only for
step-0 tokens before the first retrieval. Other tokens default to `0.0` unless a
previous `<information>...</information>` block provides an explicit hit/miss
value.

The plain OPD diagnostics support this direction:

- 25,088 retained trajectories had retrieval hit rate 0.620, so hit/miss states
  are both common enough for a gate to matter.
- Retrieval-miss trajectories had consistently higher teacher entropy than hit
  trajectories after search, e.g. turn 1: 0.899 vs 0.728, turn 2: 1.167 vs
  0.980, turn 4: 1.437 vs 1.336.
- Wrong trajectories also had higher teacher entropy after search, suggesting
  teacher uncertainty is tied to poor evidence states rather than only format
  noise.

Enable it with:

```bash
OPD_RCE_ENABLE=true ./train_opd.sh
```

Useful overrides:

```bash
OPD_RCE_ENABLE=true \
OPD_RCE_ENTROPY_NORMALIZATION=robust_minmax \
OPD_RCE_W_MIN=0.1 \
OPD_RCE_W_MAX=1.0 \
OPD_RCE_ALPHA=4.0 \
OPD_RCE_TAU=0.0 \
./train_opd.sh
```

Common modes:

```bash
# Plain OPD, no GRPO reward and no sigmoid gate.
./train_plain_opd.sh

# Equivalent explicit invocation.
OPD_GRPO_REWARD_COEF=0.0 OPD_USE_GATED_DISTILLATION=false OPD_N_AGENT=1 ./train_opd.sh

# OPD + GRPO reward, but no sigmoid gate.
OPD_GRPO_REWARD_COEF=1.0 OPD_USE_GATED_DISTILLATION=false OPD_N_AGENT=8 ./train_opd.sh

# Sigmoid-gated OPD + GRPO reward.
OPD_GRPO_REWARD_COEF=1.0 OPD_USE_GATED_DISTILLATION=true OPD_N_AGENT=8 ./train_opd.sh
```

The actor uses asymmetric dual-clip PPO, matching the SOD training setup.
`clip_ratio_low` and `clip_ratio_high` control the standard PPO ratio bounds.
For negative advantages, `clip_ratio_c` caps the loss contribution from an
abnormally large policy ratio. Monitor `actor/pg_clipfrac_lower` to see how
often this additional bound is active.

`opd/divergence` follows the SOD-style distribution-shift metric: the mean
absolute student-teacher log-probability gap on trainable tokens.
`opd/reverse_kl_k1` separately reports the signed sampled reverse-KL estimate.

Monitor `opd/teacher_entropy` alongside `opd/student_entropy` to compare the
teacher and student categorical entropy on the sampled trajectories. Both
metrics use the trainable-token mask when state masking is enabled.
`opd/beta` is the dynamic gate factor (`1.0` when gated distillation is off);
`opd/effective_distillation_coef` is the actual coefficient multiplying the OPD
advantage.

Start from `train_opd.sh` and set `STUDENT_MODEL`, `TEACHER_MODEL`, and
`DATA_DIR` for the local environment.
