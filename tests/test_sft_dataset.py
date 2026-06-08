import tempfile
import unittest
from pathlib import Path

import pandas as pd

from verl.utils.dataset import SFTDataset


class SimpleTokenizer:
    pad_token_id = 0
    eos_token_id = 999
    pad_token = "<pad>"
    eos_token = "<eos>"
    chat_template = "simple"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert not tokenize
        prompt = "".join(message["content"] for message in messages)
        if add_generation_prompt:
            return f"<user>{prompt}</user><assistant>"
        return f"<user>{prompt}</user>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert return_tensors == "pt"
        assert not add_special_tokens
        token_ids = [ord(ch) for ch in text]
        import torch

        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


def _masked_response(response):
    parts = []
    cursor = 0
    while True:
        start = response.find("<information>", cursor)
        if start == -1:
            parts.append((response[cursor:], 1))
            break
        if start > cursor:
            parts.append((response[cursor:start], 1))
        end = response.find("</information>", start)
        assert end != -1
        end += len("</information>")
        parts.append((response[start:end], 0))
        cursor = end
    return parts


class SFTDatasetTests(unittest.TestCase):
    def test_sft_dataset_masks_prompt_and_information_tokens(self):
        prompt = [{"role": "user", "content": "Who won?"}]
        response = (
            "<think>need evidence</think>"
            "<search>champion</search>"
            "<information>Doc 1: facts</information>"
            "<think>done</think>"
            "<answer>Team A</answer>"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "train.parquet"
            pd.DataFrame([{"prompt": prompt, "response": response}]).to_parquet(parquet_path)

            dataset = SFTDataset(
                parquet_files=str(parquet_path),
                tokenizer=SimpleTokenizer(),
                prompt_key="prompt",
                response_key="response",
                max_length=512,
                truncation="error",
            )

            item = dataset[0]
            valid_mask = item["attention_mask"].bool()
            loss_mask = item["loss_mask"][valid_mask].tolist()

        prompt_text = SimpleTokenizer().apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        expected_loss_mask = [0] * len(prompt_text)
        for text, trainable in _masked_response(response):
            expected_loss_mask.extend([trainable] * len(text))
        expected_loss_mask.append(1)

        self.assertEqual(loss_mask, expected_loss_mask)

    def test_sft_dataset_is_exported(self):
        self.assertEqual(SFTDataset.__name__, "SFTDataset")


if __name__ == "__main__":
    unittest.main()
