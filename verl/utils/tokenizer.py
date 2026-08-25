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
"""Utils for tokenization."""
import warnings

__all__ = [
    'hf_tokenizer',
    'mask_logits_to_tokenizer_vocab',
    'mask_vllm_logits_to_tokenizer_vocab',
    'validate_same_model_vocab',
    'validate_same_tokenizer_vocab',
]


def mask_logits_to_tokenizer_vocab(token_ids, logits, *, vocab_size):
    """Mask padded model-head rows while retaining tokenizer added tokens.

    vLLM's ``allowed_token_ids`` validates against ``tokenizer.vocab_size``,
    which excludes added tokens for Hugging Face tokenizers.  OPD instead uses
    ``len(tokenizer)`` as the shared coordinate space, so apply the upper-bound
    mask directly to the model logits.
    """
    del token_ids
    if vocab_size <= 0 or vocab_size > logits.shape[-1]:
        raise ValueError(
            f'shared tokenizer vocabulary {vocab_size} is incompatible with '
            f'rollout logits width {logits.shape[-1]}'
        )
    logits[..., vocab_size:] = float('-inf')
    return logits


def mask_vllm_logits_to_tokenizer_vocab(vocab_size, prompt_token_ids, token_ids, logits):
    """vLLM three-argument adapter for the shared-vocabulary mask.

    vLLM 0.6.3 decides whether to call a logits processor with two or three
    positional arguments by inspecting its signature.  Bind ``vocab_size`` as
    the leading positional argument with ``functools.partial`` so the resulting
    callable has exactly ``(prompt_token_ids, token_ids, logits)``.
    """
    del prompt_token_ids
    return mask_logits_to_tokenizer_vocab(token_ids, logits, vocab_size=vocab_size)


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer.

    Args:
        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.
        **kwargs: The keyword arguments for the tokenizer.

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def validate_same_tokenizer_vocab(student_tokenizer, teacher_tokenizer):
    """Fail unless two tokenizers have exactly aligned integer coordinates."""
    if len(student_tokenizer) != len(teacher_tokenizer):
        raise ValueError(
            'OPD requires identical tokenizer lengths: '
            f'{len(student_tokenizer)} != {len(teacher_tokenizer)}'
        )

    token_ids = list(range(len(student_tokenizer)))
    if student_tokenizer.convert_ids_to_tokens(token_ids) != teacher_tokenizer.convert_ids_to_tokens(token_ids):
        raise ValueError('OPD tokenizer id-to-token coordinates differ between student and teacher')

    if list(student_tokenizer.get_added_vocab().items()) != list(teacher_tokenizer.get_added_vocab().items()):
        raise ValueError('OPD tokenizer added-token order differs between student and teacher')

    if student_tokenizer.all_special_ids != teacher_tokenizer.all_special_ids:
        raise ValueError('OPD tokenizer special-token ids differ between student and teacher')
    if student_tokenizer.special_tokens_map != teacher_tokenizer.special_tokens_map:
        raise ValueError('OPD tokenizer special-token maps differ between student and teacher')


def validate_same_model_vocab(student_config, teacher_config, tokenizer):
    """Ensure both padded model heads cover the complete shared tokenizer V."""
    shared_vocab_size = len(tokenizer)
    for role, config in (('student', student_config), ('teacher', teacher_config)):
        if shared_vocab_size > config.vocab_size:
            raise ValueError(
                f'OPD shared tokenizer V exceeds the {role} model output: '
                f'{shared_vocab_size} > {config.vocab_size}'
            )
