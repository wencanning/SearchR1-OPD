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

import os
import re
from typing import List, Sequence, Union

import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import hf_tokenizer


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning on prompt/response trajectories.

    Prompt tokens and every ``<information>...</information>`` span are masked out so
    the loss only applies to agent-generated actions, reasoning, and answers.
    """

    info_pattern = re.compile(r"<information>.*?</information>", re.DOTALL)

    def __init__(self,
                 parquet_files: Union[str, Sequence[str]],
                 tokenizer: Union[str, PreTrainedTokenizer],
                 prompt_key='prompt',
                 response_key='response',
                 prompt_dict_keys=None,
                 response_dict_keys=None,
                 max_length=4096,
                 truncation='error',
                 add_eos_token=True,
                 cache_dir='~/.cache/verl/sft'):
        if not isinstance(parquet_files, (list, tuple, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.prompt_dict_keys = list(prompt_dict_keys) if prompt_dict_keys is not None else None
        self.response_dict_keys = list(response_dict_keys) if response_dict_keys is not None else None
        self.max_length = max_length
        self.truncation = truncation
        self.add_eos_token = add_eos_token

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files(self):
        dataframes = [pd.read_parquet(parquet_file) for parquet_file in self.parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)

    def __len__(self):
        return len(self.dataframe)

    def _resolve_value(self, row_dict, key, nested_keys=None):
        value = row_dict[key]
        if nested_keys is not None:
            for nested_key in nested_keys:
                value = value[nested_key]
        return value

    def _normalize_prompt(self, prompt):
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        return prompt

    def _prompt_to_text(self, prompt):
        prompt = self._normalize_prompt(prompt)
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        return prompt[0]['content']

    def _tokenize_text(self, text):
        tokenized = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        return tokenized['input_ids'][0]

    def _build_response_ids_and_loss_mask(self, response):
        response_parts = []
        cursor = 0
        for match in self.info_pattern.finditer(response):
            if match.start() > cursor:
                response_parts.append((response[cursor:match.start()], 1))
            response_parts.append((match.group(0), 0))
            cursor = match.end()
        if cursor < len(response):
            response_parts.append((response[cursor:], 1))

        input_chunks = []
        loss_mask_chunks = []
        for text, trainable in response_parts:
            if not text:
                continue
            token_ids = self._tokenize_text(text)
            input_chunks.append(token_ids)
            loss_mask_chunks.append(torch.full_like(token_ids, fill_value=trainable))

        if self.add_eos_token:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                input_chunks.append(torch.tensor([eos_token_id], dtype=torch.long))
                loss_mask_chunks.append(torch.ones(1, dtype=torch.long))

        if not input_chunks:
            raise ValueError('Response must contain at least one token after tokenization.')

        return torch.cat(input_chunks), torch.cat(loss_mask_chunks)

    def _truncate_or_pad(self, input_ids, loss_mask):
        attention_mask = torch.ones_like(input_ids)
        seq_len = input_ids.shape[0]

        if seq_len > self.max_length:
            if self.truncation == 'left':
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{seq_len=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')
        elif seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat(
                (input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)), dim=0)
            attention_mask = torch.cat((attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)), dim=0)
            loss_mask = torch.cat((loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)), dim=0)

        return input_ids, attention_mask, loss_mask

    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()
        prompt = self._resolve_value(row_dict, self.prompt_key, self.prompt_dict_keys)
        response = self._resolve_value(row_dict, self.response_key, self.response_dict_keys)

        prompt_text = self._prompt_to_text(prompt)
        prompt_ids = self._tokenize_text(prompt_text)
        response_ids, response_loss_mask = self._build_response_ids_and_loss_mask(response)

        input_ids = torch.cat((prompt_ids, response_ids), dim=0)
        loss_mask = torch.cat((torch.zeros_like(prompt_ids), response_loss_mask), dim=0)
        input_ids, attention_mask, loss_mask = self._truncate_or_pad(input_ids, loss_mask)
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
        }
