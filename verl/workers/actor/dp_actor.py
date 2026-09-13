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
Single Process Actor
"""

import itertools
from contextlib import contextmanager, nullcontext
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from verl.utils.evidence_residual import (
    build_evidence_hidden_attention_mask,
    compute_evidence_residual_target_stats,
    compute_temperature_target_stats,
)

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


def _unwrap_output_embedding(model: nn.Module) -> nn.Module:
    """Return the causal LM output projection without bypassing FSDP forward."""
    unwrapped = model.module if isinstance(model, FSDP) else model
    output_embedding = unwrapped.get_output_embeddings()
    if output_embedding is None:
        raise RuntimeError('teacher model does not expose an output embedding')
    return output_embedding


@contextmanager
def _select_output_embedding_rows(model: nn.Module, row_mask: torch.Tensor):
    """Project only selected sequence rows while retaining the normal model forward."""
    output_embedding = _unwrap_output_embedding(model)

    def select_rows(_module, inputs):
        hidden_states = inputs[0]
        if hidden_states.shape[:2] != row_mask.shape:
            raise RuntimeError(
                'teacher LM-head selection mask does not match hidden-state rows: '
                f'{tuple(row_mask.shape)} vs {tuple(hidden_states.shape[:2])}'
            )
        return (hidden_states[row_mask], *inputs[1:])

    handle = output_embedding.register_forward_pre_hook(select_rows)
    try:
        yield
    finally:
        handle.remove()


def _trim_shared_prompt_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    evidence_mask: torch.Tensor,
    response_length: int,
):
    """Remove leading prompt columns that are padding for the whole micro-batch."""
    prompt_length = input_ids.size(-1) - response_length
    if prompt_length <= 0:
        return input_ids, attention_mask, position_ids, evidence_mask
    valid_prompt_columns = attention_mask[:, :prompt_length].bool().any(dim=0)
    valid_indices = torch.nonzero(valid_prompt_columns, as_tuple=False)
    trim_start = int(valid_indices[0].item()) if valid_indices.numel() else prompt_length
    if trim_start == 0:
        return input_ids, attention_mask, position_ids, evidence_mask
    return (
        input_ids[:, trim_start:],
        attention_mask[:, trim_start:],
        position_ids[:, trim_start:],
        evidence_mask[:, trim_start:],
    )


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        self.logit_vocab_size = self.config.get('logit_vocab_size')
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                if self.logit_vocab_size is not None:
                    logits_rmpad = logits_rmpad[..., :self.logit_vocab_size]

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                if self.logit_vocab_size is not None:
                    logits = logits[..., :self.logit_vocab_size]
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto, return_entropy: bool = False):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        entropy_lst = []
        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            if return_entropy:
                entropy_lst.append(entropy)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0) if return_entropy else None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if return_entropy:
                entropy = entropy[revert_indices]

        if return_entropy:
            return log_probs, entropy
        return log_probs

    def _forward_padded_teacher_logits(
        self,
        input_ids,
        position_ids,
        attention_mask,
        response_length,
        target_row_mask,
    ):
        """Return raw teacher logits only at rows predicting policy tokens."""
        target_forward_dtype = self.config.get('target_forward_dtype', 'bfloat16')
        if target_forward_dtype in ('fp32', 'float32', 32, '32'):
            forward_context = nullcontext()
        elif target_forward_dtype in ('bf16', 'bfloat16'):
            forward_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        elif target_forward_dtype in ('fp16', 'float16', 16, '16'):
            forward_context = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            raise ValueError(f'unsupported target_forward_dtype: {target_forward_dtype}')

        select_policy_logits = self.config.get('target_select_policy_logits', True)
        if select_policy_logits:
            projection_row_mask = torch.cat(
                [
                    target_row_mask.bool(),
                    torch.zeros(
                        (target_row_mask.size(0), 1),
                        dtype=torch.bool,
                        device=target_row_mask.device,
                    ),
                ],
                dim=-1,
            )
            projection_context = _select_output_embedding_rows(
                self.actor_module,
                projection_row_mask,
            )
        else:
            projection_context = nullcontext()

        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(self.config.get('allow_tf32', False))
        try:
            with forward_context, projection_context:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    num_logits_to_keep=response_length + 1,
                )
                if select_policy_logits:
                    response_logits = output.logits[:, :self.logit_vocab_size].contiguous()
                else:
                    response_logits = output.logits[
                        :, -response_length - 1:-1, :self.logit_vocab_size
                    ].contiguous()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        if target_forward_dtype in ('fp32', 'float32', 32, '32') and response_logits.dtype != torch.float32:
            raise RuntimeError(
                'strict intervened targets require raw teacher logits to be produced in FP32; '
                f'got {response_logits.dtype}'
            )
        if target_forward_dtype in ('bf16', 'bfloat16') and response_logits.dtype != torch.bfloat16:
            raise RuntimeError(f'BF16 teacher forward produced {response_logits.dtype} logits')
        if target_forward_dtype in ('fp16', 'float16', 16, '16') and response_logits.dtype != torch.float16:
            raise RuntimeError(f'FP16 teacher forward produced {response_logits.dtype} logits')
        del output
        return response_logits

    def _forward_teacher_target_micro_batch(self,
                                            micro_batch,
                                            target_mode: str,
                                            entropy_matched_tau: float,
                                            token_chunk_size: int,
                                            evidence_residual_alpha: float = 1.0):
        response_length = micro_batch['responses'].size(-1)
        target_row_mask = micro_batch['loss_mask'].bool()
        if not torch.any(target_row_mask):
            raise ValueError('teacher target micro-batch contains no policy-token rows')
        input_ids = micro_batch['input_ids']
        attention_mask = micro_batch['attention_mask']
        position_ids = micro_batch['position_ids']
        evidence_mask = (
            micro_batch['evidence_mask']
            if 'evidence_mask' in micro_batch
            else torch.zeros_like(attention_mask)
        )
        original_sequence_length = input_ids.size(-1)
        if self.config.get('target_trim_shared_prompt_padding', True):
            input_ids, attention_mask, position_ids, evidence_mask = _trim_shared_prompt_padding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                evidence_mask=evidence_mask,
                response_length=response_length,
            )
        if not getattr(self, '_reported_teacher_target_execution', False):
            print(
                'Teacher target execution: '
                f'forward_dtype={self.config.get("target_forward_dtype", "bfloat16")}, '
                f'allow_tf32={bool(self.config.get("allow_tf32", False))}, '
                f'select_policy_logits={bool(self.config.get("target_select_policy_logits", True))}, '
                f'sequence_length={original_sequence_length}->{input_ids.size(-1)}, '
                f'lm_head_rows={int(target_row_mask.sum().item())}/'
                f'{target_row_mask.numel()}'
            )
            self._reported_teacher_target_execution = True
        observed_logits = self._forward_padded_teacher_logits(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            response_length=response_length,
            target_row_mask=target_row_mask,
        )
        if observed_logits.ndim == 3:
            observed_logits = observed_logits[target_row_mask]
        target_labels = micro_batch['responses'][target_row_mask]

        def restore_response_shape(values):
            restored = torch.zeros(
                micro_batch['responses'].shape,
                dtype=values.dtype,
                device=values.device,
            )
            restored[target_row_mask] = values
            return restored

        if target_mode == 'observed':
            # This is mathematically the ordinary teacher distribution. Unlike
            # compute_log_prob's remove-padding path, it retains the per-example
            # attention mask and normalizes selected FP16/BF16 logits in FP32.
            stats = compute_temperature_target_stats(
                observed_logits=observed_logits,
                labels=target_labels,
                temperature=1.0,
                token_chunk_size=token_chunk_size,
            )
            output = {
                'ref_log_prob': restore_response_shape(stats['target_log_prob']),
                'ref_entropy': restore_response_shape(stats['target_entropy']),
            }
        elif target_mode == 'evidence_residual':
            hidden_attention_mask = build_evidence_hidden_attention_mask(
                attention_mask=attention_mask,
                evidence_mask=evidence_mask,
                dtype=observed_logits.dtype,
            )
            hidden_logits = self._forward_padded_teacher_logits(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=hidden_attention_mask,
                response_length=response_length,
                target_row_mask=target_row_mask,
            )
            if hidden_logits.ndim == 3:
                hidden_logits = hidden_logits[target_row_mask]
            stats = compute_evidence_residual_target_stats(
                observed_logits=observed_logits,
                hidden_logits=hidden_logits,
                labels=target_labels,
                token_chunk_size=token_chunk_size,
                alpha=evidence_residual_alpha,
            )
            del hidden_logits
            output = {
                'ref_log_prob': restore_response_shape(stats['target_log_prob']),
                'ref_entropy': restore_response_shape(stats['target_entropy']),
                'ref_observed_log_prob': restore_response_shape(stats['observed_log_prob']),
                'ref_observed_entropy': restore_response_shape(stats['observed_entropy']),
                'ref_hidden_log_prob': restore_response_shape(stats['hidden_log_prob']),
                'ref_hidden_entropy': restore_response_shape(stats['hidden_entropy']),
                'ref_selected_logit_delta': restore_response_shape(stats['selected_logit_delta']),
            }
        elif target_mode == 'entropy_matched':
            stats = compute_temperature_target_stats(
                observed_logits=observed_logits,
                labels=target_labels,
                temperature=entropy_matched_tau,
                token_chunk_size=token_chunk_size,
            )
            output = {
                'ref_log_prob': restore_response_shape(stats['target_log_prob']),
                'ref_entropy': restore_response_shape(stats['target_entropy']),
            }
        else:
            raise ValueError(f'Unsupported teacher target mode: {target_mode}')

        del observed_logits
        return output

    def compute_teacher_target_log_prob(self,
                                        data: DataProto,
                                        target_mode: str,
                                        entropy_matched_tau: float = None,
                                        token_chunk_size: int = 16,
                                        evidence_residual_alpha: float = 1.0):
        """Score an observed, evidence-residual, or entropy-matched target."""
        if self.use_ulysses_sp:
            raise ValueError('intervened teacher targets require ulysses_sequence_parallel_size=1')
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'loss_mask']
        if target_mode == 'evidence_residual':
            select_keys.append('evidence_mask')
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        output_lists = {}
        for micro_batch in micro_batches:
            with torch.no_grad():
                micro_output = self._forward_teacher_target_micro_batch(
                    micro_batch=micro_batch,
                    target_mode=target_mode,
                    entropy_matched_tau=entropy_matched_tau,
                    token_chunk_size=token_chunk_size,
                    evidence_residual_alpha=evidence_residual_alpha,
                )
            for key, value in micro_output.items():
                output_lists.setdefault(key, []).append(value)

        outputs = {key: torch.cat(values, dim=0) for key, values in output_lists.items()}
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            first_output = next(iter(outputs.values()))
            assert len(indices) == first_output.size(0), f'{len(indices)} vs. {first_output.size()}'
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
                device=first_output.device,
            )
            outputs = {key: value[revert_indices] for key, value in outputs.items()}
        return outputs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        token_level_loss_normalization = bool(
            data.meta_info.get('token_level_loss_normalization', False)
        )

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            global_loss_token_count = None
            if token_level_loss_normalization:
                mini_response_length = mini_batch['responses'].size(-1)
                mini_response_mask = (
                    mini_batch['loss_mask']
                    if self.config.state_masking
                    else mini_batch['attention_mask'][:, -mini_response_length:]
                )
                global_loss_token_count = mini_response_mask.sum().float()
                torch.distributed.all_reduce(
                    global_loss_token_count,
                    op=torch.distributed.ReduceOp.SUM,
                )
                if global_loss_token_count.item() <= 0:
                    raise ValueError('intervened OPD mini-batch contains no trainable policy tokens')
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = self.config.get('clip_ratio_low', None)
                clip_ratio_high = self.config.get('clip_ratio_high', None)
                clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                )
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                if token_level_loss_normalization:
                    # FSDP averages gradients across ranks.  Scaling each local
                    # token sum by world_size/global_count recovers one exact
                    # mean over all valid policy tokens, independent of how
                    # trajectories are split across micro-batches and ranks.
                    local_loss_token_count = response_mask.sum().float()
                    loss = (
                        policy_loss
                        * local_loss_token_count
                        * torch.distributed.get_world_size()
                        / global_loss_token_count
                    )
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                if token_level_loss_normalization:
                    data['actor/global_loss_token_count'] = global_loss_token_count.detach().item()
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
