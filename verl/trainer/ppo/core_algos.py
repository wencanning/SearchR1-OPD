# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_opd_advantage(old_log_prob: torch.Tensor,
                          teacher_log_prob: torch.Tensor,
                          eos_mask: torch.Tensor,
                          advantage_mode: str = 'token',
                          normalize: bool = False,
                          clip_value: float = None):
    """Compute the K1 sampled reverse-KL signal for on-policy distillation.

    The student samples trajectories on-policy and the teacher scores the same
    tokens. ``teacher_log_prob - old_log_prob`` is a reward whose policy
    gradient minimizes ``KL(student || teacher)``.
    """
    with torch.no_grad():
        token_advantages = (teacher_log_prob - old_log_prob) * eos_mask

        if advantage_mode == 'token':
            advantages = token_advantages
        elif advantage_mode == 'sequence':
            advantages = torch.flip(
                torch.cumsum(torch.flip(token_advantages, dims=[-1]), dim=-1),
                dims=[-1],
            )
        else:
            raise ValueError(f"Unsupported OPD advantage_mode: {advantage_mode}")

        if clip_value is not None:
            advantages = torch.clamp(advantages, min=-clip_value, max=clip_value)
        if normalize:
            advantages = verl_F.masked_whiten(advantages, eos_mask)

        advantages = advantages * eos_mask

    return advantages, advantages


def compute_sod_stepwise_weights(old_log_prob: torch.Tensor,
                                 teacher_log_prob: torch.Tensor,
                                 step_mask: torch.Tensor,
                                 epsilon: float = 1e-6,
                                 delta: float = 0.2):
    """Compute the SOD weight for every policy token.

    This is adapted from ``YoungZ365/SOD`` commit ``110c4b8`` for the
    Search-R1 rollout representation.  A step is one contiguous run in
    ``step_mask``; Search-R1's observation tokens are zero in that mask and
    therefore separate consecutive assistant turns.

    For step ``k``, ``d_k`` is the mean absolute sampled-token log-probability
    gap between the student and teacher.  The first step has weight one and
    later steps use the cumulative adjacent-divergence ratio from SOD:

    ``w_k = min(prod_{u < k} (d_u + epsilon) / (d_{u+1} + epsilon), 1 + delta)``.

    Returns the token-broadcast step weights and divergences.  Both tensors are
    zero outside policy steps.
    """
    if old_log_prob.shape != teacher_log_prob.shape:
        raise ValueError('student and teacher log probabilities must have the same shape')
    if step_mask.shape != old_log_prob.shape:
        raise ValueError('step_mask must have the same shape as log probabilities')
    if old_log_prob.dim() != 2:
        raise ValueError('SOD expects rank-2 (batch, response_length) tensors')
    if epsilon <= 0:
        raise ValueError(f'SOD epsilon must be positive, got {epsilon}')
    if delta < 0:
        raise ValueError(f'SOD delta must be non-negative, got {delta}')

    with torch.no_grad():
        policy_mask = step_mask.bool()
        abs_logprob_gap = (old_log_prob - teacher_log_prob).abs().float()
        stepwise_weights = torch.zeros_like(abs_logprob_gap, dtype=torch.float32)
        stepwise_divergence = torch.zeros_like(abs_logprob_gap, dtype=torch.float32)
        upper_bound = 1.0 + float(delta)

        for batch_idx in range(policy_mask.shape[0]):
            active_indices = torch.nonzero(
                policy_mask[batch_idx], as_tuple=False
            ).squeeze(-1)
            if active_indices.numel() == 0:
                continue

            split_locations = torch.nonzero(
                active_indices[1:] != active_indices[:-1] + 1,
                as_tuple=False,
            ).squeeze(-1)
            segment_starts = torch.cat((
                active_indices.new_tensor([0]),
                split_locations + 1,
            ))
            segment_ends = torch.cat((
                split_locations + 1,
                active_indices.new_tensor([active_indices.numel()]),
            ))

            divergences = []
            boundaries = []
            for start_offset, end_offset in zip(segment_starts, segment_ends):
                segment_indices = active_indices[start_offset:end_offset]
                start = int(segment_indices[0].item())
                end = int(segment_indices[-1].item()) + 1
                divergence = abs_logprob_gap[batch_idx, start:end].mean()
                divergences.append(divergence)
                boundaries.append((start, end))

            weights = [abs_logprob_gap.new_tensor(1.0)]
            cumulative_ratio = abs_logprob_gap.new_tensor(1.0)
            for step_idx in range(len(divergences) - 1):
                cumulative_ratio = cumulative_ratio * (
                    (divergences[step_idx] + epsilon)
                    / (divergences[step_idx + 1] + epsilon)
                )
                weights.append(torch.clamp(cumulative_ratio, max=upper_bound))

            for (start, end), divergence, weight in zip(
                boundaries, divergences, weights
            ):
                stepwise_weights[batch_idx, start:end] = weight
                stepwise_divergence[batch_idx, start:end] = divergence

    return stepwise_weights, stepwise_divergence


def _normalize_vector(values: torch.Tensor,
                      method: str = 'percentile_rank',
                      eps: float = 1e-8):
    normalized = torch.full_like(values, 0.5, dtype=torch.float32)
    if values.numel() == 0:
        return normalized
    values = values.float()
    if values.numel() == 1:
        return normalized
    if method == 'percentile_rank':
        sorted_values, sorted_indices = torch.sort(values)
        unique_values = torch.unique(sorted_values, sorted=True)
        if unique_values.numel() == 1:
            return normalized
        for unique_value in unique_values:
            tie_mask = sorted_values == unique_value
            tie_positions = torch.nonzero(tie_mask, as_tuple=False).squeeze(-1).float()
            normalized[sorted_indices[tie_mask]] = tie_positions.mean() / (values.numel() - 1)
    elif method == 'robust_minmax':
        if values.numel() == 1:
            return normalized
        q05 = torch.quantile(values, 0.05)
        q95 = torch.quantile(values, 0.95)
        normalized = torch.clamp((values - q05) / (q95 - q05 + eps), 0.0, 1.0)
    elif method == 'none':
        normalized = torch.clamp(values, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported entropy normalization method: {method}")
    return normalized


def _compute_sequence_values(values: torch.Tensor,
                             mask: torch.Tensor,
                             default_value: float):
    sequence_values = torch.full_like(values, default_value, dtype=torch.float32)
    valid_sequence_values = []
    valid_sequence_indices = []
    for batch_idx in range(values.shape[0]):
        sequence_mask = mask[batch_idx].bool()
        if not torch.any(sequence_mask):
            continue
        sequence_value = values[batch_idx, sequence_mask].float().mean()
        sequence_values[batch_idx, sequence_mask] = sequence_value
        valid_sequence_values.append(sequence_value)
        valid_sequence_indices.append(batch_idx)
    return sequence_values, valid_sequence_values, valid_sequence_indices


def _normalize_sequence_values(values: torch.Tensor,
                               valid_values: list,
                               valid_indices: list,
                               method: str,
                               eps: float):
    normalized = torch.full_like(values, 0.5, dtype=torch.float32)
    if not valid_values:
        return normalized
    normalized_valid = _normalize_vector(torch.stack(valid_values), method=method, eps=eps)
    for norm_value, batch_idx in zip(normalized_valid, valid_indices):
        normalized[batch_idx] = norm_value
    return normalized


def _compute_step_values_and_normalized_entropy(values: torch.Tensor,
                                                retrieval_values: torch.Tensor,
                                                mask: torch.Tensor,
                                                step_ids: torch.Tensor,
                                                entropy_normalization: str,
                                                default_retrieval_hit: float,
                                                eps: float):
    entropy_values = torch.full_like(values, 0.0, dtype=torch.float32)
    normalized_entropy = torch.full_like(values, 0.5, dtype=torch.float32)
    step_retrieval_values = torch.full_like(values, default_retrieval_hit, dtype=torch.float32)

    step_entropies = []
    step_locations = []
    valid = mask.bool() & (step_ids >= 0)
    for batch_idx in range(values.shape[0]):
        batch_valid = valid[batch_idx]
        if not torch.any(batch_valid):
            continue
        for step_id in torch.unique(step_ids[batch_idx][batch_valid]):
            step_mask = batch_valid & (step_ids[batch_idx] == step_id)
            step_entropy = values[batch_idx, step_mask].float().mean()
            step_retrieval = retrieval_values[batch_idx, step_mask].float().mean()
            entropy_values[batch_idx, step_mask] = step_entropy
            step_retrieval_values[batch_idx, step_mask] = step_retrieval
            step_entropies.append(step_entropy)
            step_locations.append((batch_idx, step_mask))

    if step_entropies:
        normalized_steps = _normalize_vector(
            torch.stack(step_entropies),
            method=entropy_normalization,
            eps=eps,
        )
        for norm_value, (batch_idx, step_mask) in zip(normalized_steps, step_locations):
            normalized_entropy[batch_idx, step_mask] = norm_value

    return entropy_values, normalized_entropy, step_retrieval_values


def _expand_retrieval_hit(retrieval_hit: torch.Tensor,
                          target: torch.Tensor,
                          default_retrieval_hit: float):
    if retrieval_hit is None:
        return torch.full_like(target, default_retrieval_hit, dtype=torch.float32)

    retrieval_hit = retrieval_hit.to(device=target.device, dtype=torch.float32)
    if retrieval_hit.shape == target.shape:
        return retrieval_hit
    if retrieval_hit.dim() == 1 and retrieval_hit.shape[0] == target.shape[0]:
        return retrieval_hit.unsqueeze(-1).expand_as(target)
    if retrieval_hit.numel() == 1:
        return retrieval_hit.reshape(1, 1).expand_as(target)

    raise ValueError(
        "retrieval_hit must be a scalar, shape (batch,), or shape (batch, response_length)"
    )


def compute_rce_opd_advantage(old_log_prob: torch.Tensor,
                              teacher_log_prob: torch.Tensor,
                              teacher_entropy: torch.Tensor,
                              eos_mask: torch.Tensor,
                              retrieval_hit: torch.Tensor = None,
                              step_ids: torch.Tensor = None,
                              advantage_mode: str = 'token',
                              normalize: bool = False,
                              clip_value: float = None,
                              entropy_normalization: str = 'percentile_rank',
                              w_min: float = 0.1,
                              w_max: float = 1.0,
                              alpha: float = 4.0,
                              tau: float = 0.0,
                              default_retrieval_hit: float = 0.5,
                              eps: float = 1e-8,
                              return_weights: bool = False):
    """Compute retrieval-conditioned entropy-gated OPD advantages.

    This is the sampled-token reverse-KL OPD signal multiplied by
    ``w = w_min + (w_max - w_min) * sigmoid(alpha * (r - H_norm - tau))``.
    When ``step_ids`` is provided, teacher entropy and retrieval reliability are
    averaged within each step and the resulting step weight is applied back to
    all tokens in that step. Without ``step_ids``, the whole response is treated
    as one step per sequence, which is useful for trajectory-level retrieval-hit
    proxies.
    """
    with torch.no_grad():
        if teacher_entropy.shape != old_log_prob.shape:
            raise ValueError("teacher_entropy must have the same shape as log probabilities")
        if eos_mask.shape != old_log_prob.shape:
            raise ValueError("eos_mask must have the same shape as log probabilities")
        if not 0.0 <= w_min <= w_max:
            raise ValueError(f"Expected 0 <= w_min <= w_max, got w_min={w_min}, w_max={w_max}")

        mask = eos_mask.float()
        retrieval_values = _expand_retrieval_hit(retrieval_hit, old_log_prob, default_retrieval_hit)

        if step_ids is None:
            entropy_values, entropy_list, entropy_indices = _compute_sequence_values(
                teacher_entropy.float(),
                mask,
                default_value=0.0,
            )
            normalized_entropy = _normalize_sequence_values(
                entropy_values,
                entropy_list,
                entropy_indices,
                method=entropy_normalization,
                eps=eps,
            )
            retrieval_values, _, _ = _compute_sequence_values(
                retrieval_values,
                mask,
                default_value=default_retrieval_hit,
            )
        else:
            if step_ids.shape != old_log_prob.shape:
                raise ValueError("step_ids must have the same shape as log probabilities")
            step_ids = step_ids.to(device=old_log_prob.device)
            _, normalized_entropy, retrieval_values = _compute_step_values_and_normalized_entropy(
                values=teacher_entropy.float(),
                retrieval_values=retrieval_values,
                mask=mask,
                step_ids=step_ids,
                entropy_normalization=entropy_normalization,
                default_retrieval_hit=default_retrieval_hit,
                eps=eps,
            )
        gate = torch.sigmoid(alpha * (retrieval_values - normalized_entropy - tau))
        rce_weights = (w_min + (w_max - w_min) * gate) * mask

        token_advantages = (teacher_log_prob - old_log_prob) * rce_weights

        if advantage_mode == 'token':
            advantages = token_advantages
        elif advantage_mode == 'sequence':
            advantages = torch.flip(
                torch.cumsum(torch.flip(token_advantages, dims=[-1]), dim=-1),
                dims=[-1],
            )
        else:
            raise ValueError(f"Unsupported OPD advantage_mode: {advantage_mode}")

        if clip_value is not None:
            advantages = torch.clamp(advantages, min=-clip_value, max=clip_value)
        if normalize:
            advantages = verl_F.masked_whiten(advantages, mask)

        advantages = advantages * mask

    if return_weights:
        return advantages, advantages, rce_weights
    return advantages, advantages


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob,
                        log_prob,
                        advantages,
                        eos_mask,
                        cliprange,
                        cliprange_low=None,
                        cliprange_high=None,
                        clip_ratio_c=3.0):
    """Compute the asymmetric dual-clip PPO policy loss.

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float, optional)
            Lower clipping range. Defaults to ``cliprange``.
        cliprange_high: (float, optional)
            Upper clipping range. Defaults to ``cliprange``.
        clip_ratio_c: (float)
            Dual-clip bound applied only when advantage is negative.

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            approximate KL between the current and old policies
        pg_clipfrac_lower: (float)
            fraction of tokens whose negative-advantage loss is dual-clipped

    """
    assert clip_ratio_c > 1.0, f'clip_ratio_c must be greater than 1.0, got {clip_ratio_c}'

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio,
        1.0 - cliprange_low,
        1.0 + cliprange_high,
    )
    clipped_pg_losses = torch.max(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)

    dual_clip_losses = torch.min(-advantages * clip_ratio_c, clipped_pg_losses)
    pg_clipfrac_lower = verl_F.masked_mean(
        (torch.gt(clipped_pg_losses, -advantages * clip_ratio_c) * (advantages < 0)).float(),
        eos_mask,
    )
    pg_losses = torch.where(advantages < 0, dual_clip_losses, clipped_pg_losses)

    pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
