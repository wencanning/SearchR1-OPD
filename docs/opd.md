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
    distillation_coef * opd_advantage
    + grpo_reward_coef * grpo_outcome_advantage
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

actor_rollout_ref:
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
group and added to the OPD advantage. Observation tokens are excluded from
both advantage components and all related metrics.

Start from `train_opd.sh` and set `STUDENT_MODEL`, `TEACHER_MODEL`, and
`DATA_DIR` for the local environment.
