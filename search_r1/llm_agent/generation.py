import torch
import re
import random
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _print_validation_step(self, step: int, responses: List[str], observations: List[str]) -> None:
        if not self.is_validation or not responses:
            return

        response = responses[0].strip()
        observation = observations[0].strip()
        print(f"\n[validation trajectory] turn={step + 1}")
        print(f"assistant: {response}")
        if observation:
            print(f"tool: {observation[:2000]}")

    def _print_validation_trajectory(self, response_ids: torch.Tensor) -> None:
        if not self.is_validation or response_ids.shape[0] == 0:
            return

        trajectory = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print("\n[validation trajectory] complete sample")
        print(trajectory)

    def _print_training_trajectories(self, response_ids: torch.Tensor) -> None:
        if self.is_validation or response_ids.shape[0] == 0:
            return

        try:
            num_samples = int(os.environ.get("SEARCH_R1_PRINT_ROLLOUTS", "0"))
        except ValueError:
            num_samples = 0
        if num_samples <= 0:
            return

        try:
            sample_every = int(os.environ.get("SEARCH_R1_PRINT_ROLLOUT_EVERY", "64"))
        except ValueError:
            sample_every = 64
        if sample_every > 1 and random.randint(1, sample_every) != 1:
            return

        try:
            max_chars = int(os.environ.get("SEARCH_R1_PRINT_ROLLOUT_CHARS", "12000"))
        except ValueError:
            max_chars = 12000

        for sample_idx in range(min(num_samples, response_ids.shape[0])):
            trajectory = self.tokenizer.decode(response_ids[sample_idx], skip_special_tokens=True)
            if max_chars > 0 and len(trajectory) > max_chars:
                trajectory = trajectory[:max_chars] + "\n...[truncated]"
            print(f"\n[train trajectory] sample={sample_idx}")
            print(trajectory)

    def _postprocess_responses(
        self,
        responses: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Keep the exact sampled prefix through the first complete action.

        The rollout backend's token IDs define the on-policy action.  Action
        boundaries are located in decoded text, but the returned training
        sequence is always a prefix of those original IDs.  In particular,
        decoded text is never re-tokenized to construct the rollout.
        """
        decoded_responses = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        processed_ids = []
        responses_str = []
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        def decode_ids(token_ids, skip_special_tokens):
            try:
                return self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                try:
                    return self.tokenizer.decode(
                        token_ids,
                        skip_special_tokens=skip_special_tokens,
                    )
                except TypeError:
                    return self.tokenizer.decode(token_ids)

        for row_idx, decoded_response in enumerate(decoded_responses):
            raw_ids = responses[row_idx].detach().cpu().tolist()
            if pad_token_id != eos_token_id:
                while raw_ids and raw_ids[-1] == pad_token_id:
                    raw_ids.pop()

            stop_candidates = [
                (decoded_response.find(stop_tag), stop_tag)
                for stop_tag in ('</search>', '</answer>')
                if stop_tag in decoded_response
            ]
            if not stop_candidates:
                responses_str.append(decoded_response)
                response_ids = raw_ids
            else:
                stop_start, stop_tag = min(stop_candidates, key=lambda item: item[0])
                responses_str.append(decoded_response[:stop_start + len(stop_tag)])

                # Find the shortest prefix of the sampled IDs whose decode
                # contains the complete closing tag.  This handles tokenizers
                # that merge across textual tag boundaries without ever
                # introducing a canonical re-tokenization into training.
                if not raw_ids or stop_tag not in decode_ids(raw_ids, True):
                    raise RuntimeError(
                        'rollout batch_decode/decode disagree on the action boundary; '
                        'refusing to replace sampled token IDs by re-tokenized text'
                    )
                low, high = 1, len(raw_ids)
                while low < high:
                    midpoint = (low + high) // 2
                    if stop_tag in decode_ids(raw_ids[:midpoint], True):
                        high = midpoint
                    else:
                        low = midpoint + 1
                response_ids = raw_ids[:low]

                # Preserve an EOS sampled directly after a final answer.  A
                # search EOS terminates only the per-turn generation and must
                # not be inserted into the composed multi-turn trajectory.
                if (
                    stop_tag == '</answer>'
                    and low < len(raw_ids)
                    and raw_ids[low] == eos_token_id
                ):
                    response_ids.append(eos_token_id)

            if response_ids != raw_ids[:len(response_ids)]:
                raise RuntimeError('postprocessed rollout is not a prefix of sampled token IDs')
            processed_ids.append(torch.tensor(
                response_ids,
                dtype=responses.dtype,
                device=responses.device,
            ))

            # Diagnostic only: measure whether decode->encode would have
            # changed the sampled path.  These canonical IDs never enter the
            # rollout or any log-probability computation.
            sampled_text = decode_ids(response_ids, False)
            canonical_ids = self.tokenizer(
                sampled_text,
                add_special_tokens=False,
            )['input_ids']
            if isinstance(canonical_ids, torch.Tensor):
                canonical_ids = canonical_ids.detach().cpu().tolist()
            if canonical_ids and isinstance(canonical_ids[0], list):
                canonical_ids = canonical_ids[0]
            self._sampled_response_count = getattr(
                self, '_sampled_response_count', 0
            ) + 1
            self._canonical_retokenization_mismatch_count = getattr(
                self, '_canonical_retokenization_mismatch_count', 0
            ) + int(response_ids != list(canonical_ids))

        responses = torch.nn.utils.rnn.pad_sequence(
            processed_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        return responses, responses_str

    def _tokenize_observation_with_evidence_mask(self, observation: str) -> Tuple[List[int], List[int]]:
        """Tokenize an observation and mark only retrieved-content tokens.

        The marker tokens remain visible to an intervened teacher.  A token is
        marked when its character offsets overlap text inside an
        ``<information>...</information>`` pair.  Keeping this annotation next
        to tokenization avoids reconstructing evidence spans from decoded text.
        """
        information_start = "<information>"
        information_end = "</information>"
        encoded = self.tokenizer(
            observation,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        observation_ids = list(encoded['input_ids'])
        offsets = list(encoded['offset_mapping'])

        content_spans = []
        cursor = 0
        while True:
            start_index = observation.find(information_start, cursor)
            if start_index < 0:
                break
            content_start = start_index + len(information_start)
            end_index = observation.find(information_end, content_start)
            if end_index < 0:
                break
            content_spans.append((content_start, end_index))
            cursor = end_index + len(information_end)

        evidence_mask = [
            int(
                token_end > token_start
                and any(token_start < span_end and token_end > span_start
                        for span_start, span_end in content_spans)
            )
            for token_start, token_end in offsets
        ]

        if len(observation_ids) <= self.config.max_obs_length:
            return observation_ids, evidence_mask

        if not content_spans:
            return (
                observation_ids[:self.config.max_obs_length],
                evidence_mask[:self.config.max_obs_length],
            )

        fixed_indices = [idx for idx, is_evidence in enumerate(evidence_mask) if not is_evidence]
        content_budget = self.config.max_obs_length - len(fixed_indices)
        if content_budget < 0:
            raise ValueError(
                "max_obs_length is too small to preserve the visible "
                "<information> and </information> boundary tokens"
            )
        evidence_indices = [idx for idx, is_evidence in enumerate(evidence_mask) if is_evidence]
        keep = set(fixed_indices + evidence_indices[:content_budget])
        kept_indices = [idx for idx in range(len(observation_ids)) if idx in keep]
        return (
            [observation_ids[idx] for idx in kept_indices],
            [evidence_mask[idx] for idx in kept_indices],
        )

    def _process_next_obs(self, next_obs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process observations and their aligned retrieved-content masks."""
        processed_obs_ids = []
        processed_evidence_masks = []

        for observation in next_obs:
            observation_ids, evidence_mask = self._tokenize_observation_with_evidence_mask(observation)
            processed_obs_ids.append(
                torch.tensor(observation_ids, dtype=torch.long)
            )
            processed_evidence_masks.append(
                torch.tensor(evidence_mask, dtype=torch.long)
            )

        if not processed_obs_ids:
            empty = torch.empty((0, 0), dtype=torch.long)
            return empty, empty

        observation_ids = torch.nn.utils.rnn.pad_sequence(
            processed_obs_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        evidence_masks = torch.nn.utils.rnn.pad_sequence(
            processed_evidence_masks,
            batch_first=True,
            padding_value=0,
        )
        return observation_ids, evidence_masks

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                prompt_evidence_mask: torch.Tensor,
                response: torch.Tensor, 
                info: torch.Tensor = None,
                info_evidence_mask: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        evidence_masks = [prompt_evidence_mask, torch.zeros_like(response)]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            if info_evidence_mask is None:
                raise ValueError('info_evidence_mask is required when info is provided')
            evidence_masks.append(info_evidence_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        concatenated_evidence_mask = torch.cat(evidence_masks, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        padded_evidence_mask = concatenated_evidence_mask.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, padded_evidence_mask

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          next_obs_evidence_mask: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask, responses_evidence_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    right_side['responses_evidence_mask'],
                    cur_responses,
                    next_obs_ids,
                    next_obs_evidence_mask,
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask, responses_evidence_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    right_side['responses_evidence_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len],
            'responses_evidence_mask': responses_evidence_mask[:, :max_len],
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        self._sampled_response_count = 0
        self._canonical_retokenization_mismatch_count = 0
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []],
            'responses_evidence_mask': initial_input_ids[:, []],
        }
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        invalid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, invalid_action = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                invalid_action_stats,
            )
            self._print_validation_step(step, responses_str, next_obs)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            invalid_action_stats += torch.tensor(invalid_action, dtype=torch.int)

            next_obs_ids, next_obs_evidence_mask = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                next_obs_evidence_mask,
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, invalid_action = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                invalid_action_stats,
                do_search=False,
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            invalid_action_stats += torch.tensor(invalid_action, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['invalid_action_stats'] = invalid_action_stats.tolist()
        meta_info['sampled_token_preservation_rate'] = 1.0
        meta_info['canonical_retokenization_mismatch_rate'] = (
            float(self._canonical_retokenization_mismatch_count)
            / max(self._sampled_response_count, 1)
        )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        self._print_validation_trajectory(final_output.batch['responses'])
        self._print_training_trajectories(final_output.batch['responses'])
        return final_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        response_evidence_mask = final_output.pop('responses_evidence_mask')
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        final_output['evidence_mask'] = torch.cat([
            torch.zeros_like(left_side['input_ids']),
            response_evidence_mask,
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        invalid_action_counts=None,
        do_search=True,
        max_invalid_action_retries: int = 1,
    ) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search, invalid_action = [], [], [], [], []
        if invalid_action_counts is None:
            invalid_action_counts = [0] * len(predictions)
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search and search_queries:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                invalid_action.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    invalid_action.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    invalid_action.append(0)
                else:
                    invalid_action.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                    invalid_count = int(invalid_action_counts[i])
                    if (not do_search) or invalid_count >= max_invalid_action_retries:
                        next_obs.append('')
                        dones.append(1)
                    else:
                        next_obs.append(f'\nMy previous action is invalid. \
	If I want to search, I should put the query between <search> and </search>. \
	If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                        dones.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search, invalid_action

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        response = requests.post(self.config.search_url, json=payload)
        response.raise_for_status()
        return response.json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
