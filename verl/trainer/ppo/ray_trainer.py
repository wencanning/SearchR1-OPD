# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import copy
import hashlib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from datetime import datetime

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.evidence_residual import (
    calibrate_entropy_matched_temperature,
    select_calibration_ids,
)

import re
from search_r1.diagnostics.opd_uncertainty import OPDUncertaintyDumper
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean
from search_r1.diagnostics.opd_uncertainty import (
    _decode_token,
    _ground_truth_targets,
    _retrieval_hit,
    infer_evidence_step_ids,
    retrieval_hits_by_information_block,
)


def compute_opd_logprob_metrics(old_log_probs: torch.Tensor,
                                ref_log_prob: torch.Tensor,
                                mask: torch.Tensor) -> dict:
    logprob_gap = old_log_probs - ref_log_prob
    divergence = masked_mean(logprob_gap.abs(), mask).item()
    reverse_kl_k1 = masked_mean(logprob_gap, mask).item()
    teacher_advantage = masked_mean(-logprob_gap * mask, mask).item()
    return {
        'opd/divergence': divergence,
        'opd/reverse_kl_k1': reverse_kl_k1,
        'opd/teacher_advantage': teacher_advantage,
    }


def _build_rce_retrieval_hit_values(evidence_step_ids, block_hits, pre_retrieval_hit: float) -> list[float]:
    """Map decoded Search-R1 step ids to token-level retrieval-hit values."""
    values = []
    for step_id in evidence_step_ids:
        if step_id == 0:
            values.append(float(pre_retrieval_hit))
        elif block_hits is not None and step_id - 1 < len(block_hits):
            values.append(float(block_hits[step_id - 1]))
        else:
            values.append(0.0)
    return values


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'opd':
        responses = data.batch['responses']
        response_length = responses.size(-1)
        response_mask = data.batch['loss_mask'] if 'loss_mask' in data.batch else \
            data.batch['attention_mask'][:, -response_length:]
        opd_config = data.meta_info['opd_config']
        rce_config = opd_config.get('rce', {})
        use_rce = rce_config.get('enable', False)
        if use_rce:
            if 'ref_entropy' not in data.batch:
                raise ValueError('RCE-OPD requires ref_entropy from the teacher/reference policy')
            retrieval_hit = data.batch['rce_retrieval_hit'] if 'rce_retrieval_hit' in data.batch else None
            step_ids = data.batch['rce_step_ids'] if 'rce_step_ids' in data.batch else None
            opd_advantages, _, rce_weights = core_algos.compute_rce_opd_advantage(
                old_log_prob=data.batch['old_log_probs'],
                teacher_log_prob=data.batch['ref_log_prob'],
                teacher_entropy=data.batch['ref_entropy'],
                eos_mask=response_mask,
                retrieval_hit=retrieval_hit,
                step_ids=step_ids,
                advantage_mode=opd_config['advantage_mode'],
                normalize=opd_config['normalize'],
                clip_value=opd_config['clip_value'],
                entropy_normalization=rce_config.get('entropy_normalization', 'percentile_rank'),
                w_min=rce_config.get('w_min', 0.1),
                w_max=rce_config.get('w_max', 1.0),
                alpha=rce_config.get('alpha', 4.0),
                tau=rce_config.get('tau', 0.0),
                default_retrieval_hit=rce_config.get('default_retrieval_hit', 0.5),
                eps=rce_config.get('eps', 1e-8),
                return_weights=True,
            )
        else:
            opd_advantages, _ = core_algos.compute_opd_advantage(
                old_log_prob=data.batch['old_log_probs'],
                teacher_log_prob=data.batch['ref_log_prob'],
                eos_mask=response_mask,
                advantage_mode=opd_config['advantage_mode'],
                normalize=opd_config['normalize'],
                clip_value=opd_config['clip_value'],
            )
            rce_weights = torch.zeros_like(opd_advantages)
        grpo_advantages = torch.zeros_like(opd_advantages)
        grpo_reward_coef = opd_config.get('grpo_reward_coef', 0.0)
        distillation_coef = opd_config.get('distillation_coef', 1.0)
        use_gated_distillation = opd_config.get('use_gated_distillation', False) and grpo_reward_coef != 0
        if grpo_reward_coef != 0:
            grpo_advantages, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch['token_level_rewards'],
                eos_mask=response_mask,
                index=data.non_tensor_batch['uid'],
            )

        if use_gated_distillation:
            gamma = opd_config.get('gamma', 1.0)
            beta_min = opd_config.get('beta_min', 0.0)
            beta_max = opd_config.get('beta_max', 0.05)
            task_advantages = masked_mean(grpo_advantages, response_mask, axis=1)
            gate = torch.sigmoid(-gamma * task_advantages)
            beta = beta_min + (beta_max - beta_min) * gate
        else:
            beta = torch.full(
                (opd_advantages.shape[0],),
                1.0,
                dtype=opd_advantages.dtype,
                device=opd_advantages.device,
            )
        effective_distillation_coef = distillation_coef * beta
        weighted_opd_advantages = effective_distillation_coef.unsqueeze(-1) * opd_advantages

        advantages = weighted_opd_advantages + grpo_reward_coef * grpo_advantages
        returns = advantages
        data.batch['opd_advantages'] = opd_advantages
        data.batch['opd_rce_weights'] = rce_weights
        data.batch['opd_beta'] = beta.unsqueeze(-1) * response_mask
        data.batch['opd_effective_distillation_coef'] = effective_distillation_coef.unsqueeze(-1) * response_mask
        data.batch['weighted_opd_advantages'] = weighted_opd_advantages
        data.batch['grpo_advantages'] = grpo_advantages
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    advantage_mask = batch.batch['loss_mask'].bool() if 'loss_mask' in batch.batch else response_mask

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, advantage_mask)
    valid_returns = torch.masked_select(returns, advantage_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, advantage_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())
    if 'invalid_action_stats' in batch.meta_info:
        metrics['env/number_of_invalid_action'] = float(
            np.array(batch.meta_info['invalid_action_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def validate_opd_teacher_target_config(config):
    """Validate invariants that keep intervened targets as the only variable."""
    opd_config = config.algorithm.opd
    target_mode = opd_config.get('teacher_target', 'observed')
    supported_modes = {'observed', 'evidence_residual', 'entropy_matched'}
    if target_mode not in supported_modes:
        raise ValueError(f'Unsupported OPD teacher_target: {target_mode}')
    if target_mode == 'observed':
        return

    if not config.do_search:
        raise ValueError(f'{target_mode} requires search-agent trajectories')
    if not config.actor_rollout_ref.actor.state_masking:
        raise ValueError(f'{target_mode} requires state_masking=true')
    if opd_config.advantage_mode != 'token':
        raise ValueError(f'{target_mode} requires advantage_mode=token for equal token coverage')
    if opd_config.normalize:
        raise ValueError(f'{target_mode} requires normalize=false')
    if opd_config.clip_value is not None:
        raise ValueError(f'{target_mode} requires clip_value=null')
    grpo_reward_coef = float(opd_config.get('grpo_reward_coef', 0.0))
    if grpo_reward_coef < 0:
        raise ValueError(f'{target_mode} requires grpo_reward_coef>=0')
    if opd_config.get('rce', {}).get('enable', False):
        raise ValueError(f'{target_mode} requires all RCE/token-weighting paths to be disabled')
    if config.actor_rollout_ref.actor.entropy_coeff != 0:
        raise ValueError(f'{target_mode} requires actor entropy_coeff=0')
    if config.actor_rollout_ref.actor.ppo_epochs != 1:
        raise ValueError(f'{target_mode} requires ppo_epochs=1')
    rollout_group_size = (
        config.actor_rollout_ref.rollout.n
        * config.actor_rollout_ref.rollout.n_agent
    )
    if grpo_reward_coef > 0:
        if rollout_group_size <= 1:
            raise ValueError(
                f'{target_mode}+GRPO requires more than one rollout per question'
            )
    else:
        if rollout_group_size != 1:
            raise ValueError(
                f'pure {target_mode} requires exactly one on-policy trajectory per question'
            )
        if config.actor_rollout_ref.actor.ppo_mini_batch_size != config.data.train_batch_size:
            raise ValueError(
                f'pure {target_mode} requires ppo_mini_batch_size=train_batch_size so the update '
                'is one global mean over all policy tokens'
            )
    if config.actor_rollout_ref.rollout.temperature != 1.0:
        raise ValueError(f'{target_mode} requires rollout.temperature=1.0 so teacher logits are unscaled')
    if not config.actor_rollout_ref.rollout.get('restrict_to_tokenizer_vocab', False):
        raise ValueError(f'{target_mode} requires rollout.restrict_to_tokenizer_vocab=true')
    if config.actor_rollout_ref.ref.attn_implementation != 'sdpa':
        raise ValueError(f'{target_mode} requires ref.attn_implementation=sdpa')
    if config.actor_rollout_ref.ref.ulysses_sequence_parallel_size != 1:
        raise ValueError(f'{target_mode} requires ref.ulysses_sequence_parallel_size=1')
    fp32_aliases = {'32', 'fp32', 'float32'}
    fp16_aliases = {'16', 'fp16', 'float16'}
    bf16_aliases = {'bf16', 'bfloat16'}
    ref_fsdp_config = config.actor_rollout_ref.ref.fsdp_config
    precision_settings = {
        'ref.fsdp_config.model_dtype': str(ref_fsdp_config.get('model_dtype')).lower(),
        'ref.fsdp_config.mixed_precision.param_dtype': str(
            ref_fsdp_config.mixed_precision.param_dtype
        ).lower(),
        'ref.target_forward_dtype': str(
            config.actor_rollout_ref.ref.get('target_forward_dtype')
        ).lower(),
    }
    allow_approximate_precision = bool(
        config.actor_rollout_ref.ref.get('allow_approximate_target_precision', False)
    )
    if allow_approximate_precision:
        precision_family = None
        for aliases in (fp16_aliases, bf16_aliases):
            if all(value in aliases for value in precision_settings.values()):
                precision_family = aliases
                break
        if precision_family is None:
            raise ValueError(
                f'{target_mode} approximate teacher precision requires model, mixed-precision, '
                f'and forward dtypes to consistently use FP16 or BF16; got {precision_settings}'
            )
    else:
        for name, value in precision_settings.items():
            if value not in fp32_aliases:
                raise ValueError(
                    f'{target_mode} requires {name}=fp32 unless '
                    'ref.allow_approximate_target_precision=true'
                )
    lambda_distill = opd_config.get('lambda_distill')
    if lambda_distill is None or lambda_distill <= 0:
        raise ValueError(f'{target_mode} requires a positive fixed lambda_distill')
    if opd_config.get('target_token_chunk_size', 0) <= 0:
        raise ValueError('target_token_chunk_size must be positive')
    if target_mode == 'entropy_matched':
        tau = opd_config.get('entropy_matched_tau')
        if tau is None or tau <= 0:
            raise ValueError('entropy_matched requires a positive pre-calibrated entropy_matched_tau')


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_opd = config.algorithm.adv_estimator == 'opd'
        self.use_rm = Role.RewardModel in role_worker_mapping and not self.use_opd
        self.ray_worker_group_cls = ray_worker_group_cls
        self.opd_diagnostics = None

        if self.use_opd and not self.config.trainer.get('val_only', False):
            assert self.use_reference_policy, 'OPD requires a teacher/reference policy'
            assert not config.actor_rollout_ref.actor.use_kl_loss, \
                'OPD uses the teacher signal directly; actor.use_kl_loss must be false'
            assert not config.do_search or config.actor_rollout_ref.actor.state_masking, \
                'Search OPD requires actor.state_masking=true to exclude observation tokens'
            validate_opd_teacher_target_config(config)
            if config.algorithm.opd.grpo_reward_coef != 0:
                group_size = config.actor_rollout_ref.rollout.n_agent * config.actor_rollout_ref.rollout.n
                assert group_size > 1, 'OPD + GRPO reward requires more than one rollout per prompt'
            if config.algorithm.opd.get('diagnostics', {}).get('enable', False):
                self.opd_diagnostics = OPDUncertaintyDumper(
                    tokenizer=self.tokenizer,
                    trainer_config=config.trainer,
                    opd_config=config.algorithm.opd,
                )

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        train_data_source = self.config.data.get('train_data_source')
        if train_data_source:
            if 'data_source' not in self.train_dataset.dataframe.columns:
                raise ValueError('data.train_data_source requires a data_source parquet column')
            self.train_dataset.dataframe = self.train_dataset.dataframe[
                self.train_dataset.dataframe['data_source'] == train_data_source
            ].copy()
            if self.train_dataset.dataframe.empty:
                raise ValueError(f'no training rows found for data_source={train_data_source!r}')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        val_data_source = self.config.data.get('val_data_source')
        if val_data_source:
            if 'data_source' not in self.val_dataset.dataframe.columns:
                raise ValueError('data.val_data_source requires a data_source parquet column')
            self.val_dataset.dataframe = self.val_dataset.dataframe[
                self.val_dataset.dataframe['data_source'] == val_data_source
            ].copy()
            if self.val_dataset.dataframe.empty:
                raise ValueError(f'no validation rows found for data_source={val_data_source!r}')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        import time
        reward_tensor_lst = []
        data_source_lst = []
        total_val_steps = len(self.val_dataloader)
        print(f'[validation] total batches: {total_val_steps}')

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for val_step, test_data in enumerate(self.val_dataloader, start=1):
                batch_start_time = time.perf_counter()
                print(f'[validation] running batch {val_step}/{total_val_steps}')
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                elapsed = time.perf_counter() - batch_start_time
                print(f'[validation] completed batch {val_step}/{total_val_steps} elapsed_s={elapsed:.3f}')
        else:
            for val_step, batch_dict in enumerate(self.val_dataloader, start=1):
                batch_start_time = time.perf_counter()
                print(f'[validation] running batch {val_step}/{total_val_steps}')
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                    elapsed = time.perf_counter() - batch_start_time
                    print(
                        f'[validation] completed batch {val_step}/{total_val_steps} '
                        f'elapsed_s={elapsed:.3f}'
                    )

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator in ['grpo', 'opd']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _score_calibration_entropy(self, calibration_batches, target_mode, tau=None):
        entropy_sum = 0.0
        observed_entropy_sum = 0.0
        token_count = 0
        for calibration_batch, pad_size in calibration_batches:
            calibration_batch.meta_info['opd_teacher_target'] = target_mode
            calibration_batch.meta_info['opd_entropy_matched_tau'] = tau
            calibration_batch.meta_info['opd_target_token_chunk_size'] = \
                self.config.algorithm.opd.get('target_token_chunk_size', 16)
            output = self.ref_policy_wg.compute_ref_log_prob(calibration_batch)
            valid_output = unpad_dataproto(output, pad_size)
            valid_batch = unpad_dataproto(calibration_batch, pad_size)
            policy_mask = valid_batch.batch['loss_mask'].float()
            entropy_sum += (valid_output.batch['ref_entropy'] * policy_mask).sum().item()
            if 'ref_observed_entropy' in valid_output.batch:
                observed_entropy_sum += (
                    valid_output.batch['ref_observed_entropy'] * policy_mask
                ).sum().item()
            token_count += int(policy_mask.sum().item())
        if token_count == 0:
            raise ValueError('entropy calibration collected no policy-token rows')
        return {
            'mean_entropy': entropy_sum / token_count,
            'mean_observed_entropy': observed_entropy_sum / token_count,
            'token_count': token_count,
        }

    def _run_er_entropy_calibration(self, generation_manager):
        """Collect frozen-student rows once and fit one auditable global tau."""
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import collate_fn

        calibration_config = self.config.trainer.er_entropy_calibration
        if self.config.algorithm.opd.teacher_target != 'evidence_residual':
            raise ValueError('ER entropy calibration must run with teacher_target=evidence_residual')
        if self.config.actor_rollout_ref.rollout.seed != calibration_config.decode_seed:
            raise ValueError('rollout.seed must equal er_entropy_calibration.decode_seed')

        dataframe = self.train_dataset.dataframe
        data_source = calibration_config.get('data_source')
        if data_source:
            dataframe = dataframe[dataframe['data_source'] == data_source]
        if dataframe.empty:
            raise ValueError(f'no training questions found for calibration data_source={data_source!r}')
        if 'id' not in dataframe.columns:
            raise ValueError('entropy calibration requires a stable question id column')

        selected_ids = select_calibration_ids(
            dataframe['id'].astype(str).tolist(),
            seed=calibration_config.split_seed,
            fraction=calibration_config.fraction,
        )
        selection_order = {question_id: idx for idx, question_id in enumerate(selected_ids)}
        selected_dataframe = dataframe[
            dataframe['id'].astype(str).isin(selection_order)
        ].copy()
        selected_dataframe['_calibration_order'] = selected_dataframe['id'].astype(str).map(selection_order)
        selected_dataframe = selected_dataframe.sort_values('_calibration_order').drop(
            columns=['_calibration_order'])

        calibration_dataset = copy.copy(self.train_dataset)
        calibration_dataset.dataframe = selected_dataframe
        calibration_loader = DataLoader(
            dataset=calibration_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        calibration_batches = []
        trajectory_count = 0
        for batch_dict in calibration_loader:
            batch = DataProto.from_single_dict(batch_dict)
            trajectory_count += len(batch)
            batch.non_tensor_batch['uid'] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch))],
                dtype=object,
            )
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            first_input_ids = gen_batch.batch['input_ids'][
                :, -self.config.data.max_start_length:
            ].clone().long()
            final_output = generation_manager.run_llm_loop(
                gen_batch=gen_batch,
                initial_input_ids=first_input_ids,
            )
            for key in final_output.batch.keys():
                final_output.batch[key] = final_output.batch[key].long()
            batch = batch.union(final_output)
            batch, _ = self._create_loss_mask(batch, {})
            scoring_batch = DataProto.from_dict(tensors={
                key: batch.batch[key]
                for key in (
                    'responses',
                    'input_ids',
                    'attention_mask',
                    'position_ids',
                    'evidence_mask',
                    'loss_mask',
                )
            })
            scoring_batch, pad_size = pad_dataproto_to_divisor(
                scoring_batch,
                self.ref_policy_wg.world_size,
            )
            calibration_batches.append((scoring_batch, pad_size))

        er_stats = self._score_calibration_entropy(
            calibration_batches,
            target_mode='evidence_residual',
        )

        def mean_visible_entropy(temperature):
            return self._score_calibration_entropy(
                calibration_batches,
                target_mode='entropy_matched',
                tau=temperature,
            )['mean_entropy']

        result = calibrate_entropy_matched_temperature(
            mean_visible_entropy,
            target_entropy=er_stats['mean_entropy'],
            entropy_tolerance=calibration_config.entropy_tolerance,
            relative_temperature_tolerance=calibration_config.relative_temperature_tolerance,
        )
        fingerprint_payload = {
            'selected_question_ids': selected_ids,
            'split_seed': int(calibration_config.split_seed),
            'decode_seed': int(calibration_config.decode_seed),
            'student_model': str(self.config.actor_rollout_ref.model.path),
            'teacher_model': str(self.config.actor_rollout_ref.ref.model_path),
            'shared_vocab_size': len(self.tokenizer),
        }
        calibration_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()
        artifact = {
            'schema_version': 1,
            'created_at': datetime.now().astimezone().isoformat(),
            'method': 'ER-OPD entropy-matched control calibration',
            'student_model': str(self.config.actor_rollout_ref.model.path),
            'teacher_model': str(self.config.actor_rollout_ref.ref.model_path),
            'shared_vocab_size': len(self.tokenizer),
            'data_source': data_source,
            'fraction': float(calibration_config.fraction),
            'split_seed': int(calibration_config.split_seed),
            'decode_seed': int(calibration_config.decode_seed),
            'selection_rule': 'seeded crc32 rank of question ids with lexical tie-break',
            'calibration_fingerprint_sha256': calibration_fingerprint,
            'selected_question_ids': selected_ids,
            'selected_question_count': len(selected_ids),
            'trajectory_count': trajectory_count,
            'policy_token_count': er_stats['token_count'],
            'er_mean_entropy': er_stats['mean_entropy'],
            'visible_mean_entropy_at_tau_1': er_stats['mean_observed_entropy'],
            'temperature': result.temperature,
            'matched_mean_entropy': result.achieved_entropy,
            'entropy_gap': result.entropy_gap,
            'entropy_tolerance': float(calibration_config.entropy_tolerance),
            'relative_temperature_tolerance': float(
                calibration_config.relative_temperature_tolerance),
            'solver_evaluations': result.evaluations,
            'stop_reason': result.stop_reason,
        }

        output_path = os.path.abspath(os.path.expanduser(calibration_config.output_path))
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        stem, extension = os.path.splitext(output_path)
        timestamped_path = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
        for path in (timestamped_path, output_path):
            with open(path, 'w', encoding='utf-8') as output_file:
                json.dump(artifact, output_file, ensure_ascii=False, indent=2)
                output_file.write('\n')
        print(f'ER entropy calibration saved: {timestamped_path}')
        print(f'Use ENTROPY_MATCHED_TAU={result.temperature:.12g}')
        return artifact

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _create_rce_metadata(self, batch: DataProto, metrics: dict):
        """Create token-aligned RCE metadata from decoded Search-R1 trajectories."""
        response_length = batch.batch['responses'].size(-1)
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        rce_config = self.config.algorithm.opd.get('rce', {})
        pre_retrieval_hit = float(rce_config.get('default_retrieval_hit', 0.5))

        step_ids = torch.full_like(batch.batch['responses'], -1, dtype=torch.long)
        retrieval_hit = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
        sequence_retrieval_hit = torch.full(
            (batch.batch['responses'].shape[0],),
            0.0,
            dtype=torch.float32,
        )
        retrieval_hit_known = torch.zeros_like(sequence_retrieval_hit, dtype=torch.bool)
        reward_models = batch.non_tensor_batch.get("reward_model", [None] * len(batch))

        for seq_idx in range(len(batch)):
            valid_len = int(response_mask[seq_idx].sum().item())
            response_ids = batch.batch["responses"][seq_idx, :valid_len].detach().cpu().tolist()
            token_texts = [_decode_token(self.tokenizer, int(token_id)) for token_id in response_ids]
            evidence_step_ids = infer_evidence_step_ids(token_texts)
            if evidence_step_ids:
                step_ids[seq_idx, :valid_len] = torch.tensor(
                    evidence_step_ids,
                    dtype=torch.long,
                    device=step_ids.device,
                )

            targets = _ground_truth_targets(reward_models[seq_idx])
            decoded_response = "".join(token_texts)
            sequence_hit = _retrieval_hit(decoded_response, targets)
            block_hits = retrieval_hits_by_information_block(decoded_response, targets)
            if evidence_step_ids:
                hit_values = _build_rce_retrieval_hit_values(
                    evidence_step_ids,
                    block_hits,
                    pre_retrieval_hit,
                )
                retrieval_hit[seq_idx, :valid_len] = torch.tensor(
                    hit_values,
                    dtype=torch.float32,
                    device=retrieval_hit.device,
                )
            if sequence_hit is not None:
                sequence_retrieval_hit[seq_idx] = float(sequence_hit)
                retrieval_hit_known[seq_idx] = True

        batch.batch['rce_step_ids'] = step_ids
        batch.batch['rce_retrieval_hit'] = retrieval_hit

        known_count = retrieval_hit_known.sum().item()
        if known_count > 0:
            metrics['opd/rce_retrieval_hit_proxy'] = sequence_retrieval_hit[retrieval_hit_known].mean().item()
            metrics['opd/rce_retrieval_hit_known'] = float(known_count)
        step_counts = []
        for seq_idx in range(len(batch)):
            valid_steps = step_ids[seq_idx][response_mask[seq_idx].bool() & (step_ids[seq_idx] >= 0)]
            step_counts.append(float(torch.unique(valid_steps).numel()) if valid_steps.numel() > 0 else 0.0)
        metrics['opd/rce_step_count_mean'] = float(np.mean(step_counts)) if step_counts else 0.0

        return batch, metrics

    @staticmethod
    def _plain_config(config):
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
        return dict(config) if isinstance(config, dict) else {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        if self.config.trainer.er_entropy_calibration.get('enable', False):
            self._run_er_entropy_calibration(generation_manager)
            return

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                if self.config.do_search:
                    # Keep all rollouts from the same prompt in one GRPO group.
                    # Dataset indices are not globally unique across merged sources.
                    batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            if self.use_opd or (
                                self.opd_diagnostics is not None
                                and self.opd_diagnostics.should_dump(self.global_steps)
                            ):
                                final_gen_batch_output.meta_info['return_entropy'] = True
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    float_batch_keys = {'old_log_probs', 'old_entropy'}
                    for key in batch.batch.keys():
                        if key not in float_batch_keys:
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_opd and self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                        batch, metrics = self._create_loss_mask(batch, metrics)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            if self.use_opd:
                                teacher_target = self.config.algorithm.opd.get('teacher_target', 'observed')
                                if teacher_target == 'evidence_residual' and 'evidence_mask' not in batch.batch:
                                    raise ValueError('evidence_residual target requires rollout evidence_mask')
                                batch.meta_info['opd_teacher_target'] = teacher_target
                                batch.meta_info['opd_entropy_matched_tau'] = \
                                    self.config.algorithm.opd.get('entropy_matched_tau')
                                batch.meta_info['opd_target_token_chunk_size'] = \
                                    self.config.algorithm.opd.get('target_token_chunk_size', 16)
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    if self.use_opd and self.config.algorithm.opd.get('rce', {}).get('enable', False):
                        batch, metrics = self._create_rce_metadata(batch, metrics)

                    with _timer('adv', timing_raw):
                        if self.use_opd:
                            response_length = batch.batch['responses'].size(-1)
                            opd_mask = batch.batch['loss_mask'] if 'loss_mask' in batch.batch else \
                                batch.batch['attention_mask'][:, -response_length:]
                            opd_scores = (batch.batch['ref_log_prob'] - batch.batch['old_log_probs']) * opd_mask
                            if self.config.algorithm.opd.grpo_reward_coef != 0:
                                reward_tensor = self.reward_fn(batch)
                                batch.batch['token_level_scores'] = reward_tensor
                                batch.batch['token_level_rewards'] = reward_tensor
                            else:
                                # Keep rule/EM scores for logging only. Pure OPD training
                                # still uses the teacher-student log-prob signal below.
                                reward_tensor = self.reward_fn(batch)
                                batch.batch['token_level_scores'] = reward_tensor
                                batch.batch['token_level_rewards'] = opd_scores
                            lambda_distill = self.config.algorithm.opd.get('lambda_distill')
                            if lambda_distill is None:
                                lambda_distill = self.config.algorithm.opd.distillation_coef
                            batch.meta_info['opd_config'] = {
                                'advantage_mode': self.config.algorithm.opd.advantage_mode,
                                'normalize': self.config.algorithm.opd.normalize,
                                'clip_value': self.config.algorithm.opd.clip_value,
                                'distillation_coef': lambda_distill,
                                'lambda_distill': lambda_distill,
                                'teacher_target': self.config.algorithm.opd.get('teacher_target', 'observed'),
                                'grpo_reward_coef': self.config.algorithm.opd.grpo_reward_coef,
                                'use_gated_distillation': self.config.algorithm.opd.get('use_gated_distillation', True),
                                'gamma': self.config.algorithm.opd.get('gamma', 1.0),
                                'beta_min': self.config.algorithm.opd.get('beta_min', 0.0),
                                'beta_max': self.config.algorithm.opd.get('beta_max', 0.05),
                                'rce': self._plain_config(self.config.algorithm.opd.get('rce', {})),
                            }
                            metrics.update(
                                compute_opd_logprob_metrics(
                                    old_log_probs=batch.batch['old_log_probs'],
                                    ref_log_prob=batch.batch['ref_log_prob'],
                                    mask=opd_mask,
                                ))
                            metrics['opd/teacher_entropy'] = masked_mean(
                                batch.batch['ref_entropy'],
                                opd_mask,
                            ).item()
                            teacher_target = self.config.algorithm.opd.get('teacher_target', 'observed')
                            metrics['opd/teacher_target_is_intervened'] = float(
                                teacher_target != 'observed')
                            if teacher_target == 'entropy_matched':
                                metrics['opd/entropy_matched_tau'] = float(
                                    self.config.algorithm.opd.entropy_matched_tau)
                            if teacher_target == 'evidence_residual':
                                evidence_mask = batch.batch['evidence_mask'].float()
                                valid_context = batch.batch['attention_mask'].float()
                                metrics['opd/evidence_context_token_fraction'] = (
                                    evidence_mask.sum() / valid_context.sum()
                                ).item()
                                metrics['opd/evidence_trajectory_rate'] = (
                                    evidence_mask.bool().any(dim=-1).float().mean()
                                ).item()
                                metrics['opd/observed_teacher_entropy'] = masked_mean(
                                    batch.batch['ref_observed_entropy'], opd_mask).item()
                                metrics['opd/hidden_teacher_entropy'] = masked_mean(
                                    batch.batch['ref_hidden_entropy'], opd_mask).item()
                                metrics['opd/er_logprob_shift'] = masked_mean(
                                    batch.batch['ref_log_prob'] - batch.batch['ref_observed_log_prob'],
                                    opd_mask,
                                ).item()
                                metrics['opd/selected_logit_residual'] = masked_mean(
                                    batch.batch['ref_selected_logit_delta'], opd_mask).item()
                            if 'old_entropy' in batch.batch:
                                metrics['opd/student_entropy'] = masked_mean(
                                    batch.batch['old_entropy'],
                                    opd_mask,
                                ).item()
                        else:
                            # compute scores. Support both model and function-based.
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor

                            if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                batch, kl_metrics = apply_kl_penalty(batch,
                                                                     kl_ctrl=self.kl_ctrl,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        if self.use_opd:
                            metrics['opd/distillation_advantage'] = masked_mean(
                                batch.batch['opd_advantages'], opd_mask
                            ).item()
                            metrics['opd/weighted_distillation_advantage'] = masked_mean(
                                batch.batch['weighted_opd_advantages'], opd_mask
                            ).item()
                            metrics['opd/beta'] = masked_mean(
                                batch.batch['opd_beta'], opd_mask
                            ).item()
                            metrics['opd/effective_distillation_coef'] = masked_mean(
                                batch.batch['opd_effective_distillation_coef'], opd_mask
                            ).item()
                            if self.config.algorithm.opd.get('rce', {}).get('enable', False):
                                metrics['opd/rce_weight'] = masked_mean(
                                    batch.batch['opd_rce_weights'], opd_mask
                                ).item()
                                if 'rce_retrieval_hit' in batch.batch:
                                    rce_retrieval_hit = batch.batch['rce_retrieval_hit'].float()
                                    rce_known_mask = (
                                        ((rce_retrieval_hit == 0.0) | (rce_retrieval_hit == 1.0)).float()
                                        * opd_mask
                                    )
                                    hit_mask = (rce_retrieval_hit > 0.5).float() * rce_known_mask
                                    miss_mask = (rce_retrieval_hit < 0.5).float() * rce_known_mask
                                    known_tokens = rce_known_mask.sum()
                                    if known_tokens.item() > 0:
                                        metrics['opd/rce_known_token_fraction'] = (
                                            known_tokens / opd_mask.sum()
                                        ).item()
                                        metrics['opd/rce_token_hit_rate'] = (
                                            hit_mask.sum() / known_tokens
                                        ).item()
                                    if hit_mask.sum().item() > 0:
                                        metrics['opd/rce_weight_hit'] = masked_mean(
                                            batch.batch['opd_rce_weights'], hit_mask
                                        ).item()
                                    if miss_mask.sum().item() > 0:
                                        metrics['opd/rce_weight_miss'] = masked_mean(
                                            batch.batch['opd_rce_weights'], miss_mask
                                        ).item()
                                    if 'opd/rce_weight_hit' in metrics and 'opd/rce_weight_miss' in metrics:
                                        metrics['opd/rce_weight_delta_hit_minus_miss'] = (
                                            metrics['opd/rce_weight_hit'] - metrics['opd/rce_weight_miss']
                                        )
                            metrics['opd/grpo_advantage'] = masked_mean(
                                batch.batch['grpo_advantages'], opd_mask
                            ).item()
                            metrics['opd/combined_advantage'] = masked_mean(
                                batch.batch['advantages'], opd_mask
                            ).item()
                            if self.opd_diagnostics is not None and self.opd_diagnostics.should_dump(self.global_steps):
                                with _timer('opd_diagnostics', timing_raw):
                                    diagnostics_path = self.opd_diagnostics.dump(
                                        batch=batch,
                                        opd_mask=opd_mask,
                                        global_step=self.global_steps,
                                        epoch=epoch,
                                        metrics=metrics,
                                    )
                                metrics['opd_diagnostics/trajectories'] = float(
                                    min(self.opd_diagnostics.max_sequences_per_step, len(batch)))
                                print(f'OPD diagnostics saved: {diagnostics_path}')

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if not self.use_opd and self.config.do_search and \
                                    self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            batch.meta_info['token_level_loss_normalization'] = bool(
                                self.use_opd
                                and self.config.algorithm.opd.get('teacher_target', 'observed') != 'observed'
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        if self.use_opd and 'actor/entropy_loss' in actor_output_metrics:
                            entropy_key = (
                                'opd/update_student_entropy'
                                if 'opd/student_entropy' in metrics
                                else 'opd/student_entropy'
                            )
                            actor_output_metrics[entropy_key] = actor_output_metrics.pop('actor/entropy_loss')
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
