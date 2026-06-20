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
