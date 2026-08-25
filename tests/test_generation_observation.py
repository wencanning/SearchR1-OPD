import unittest
from types import SimpleNamespace

import torch

from search_r1.llm_agent.generation import LLMGenerationManager


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        output = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            output["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return output

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids if token_id != self.pad_token_id)


class ActionTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def batch_decode(self, responses, skip_special_tokens=True):
        del responses, skip_special_tokens
        return [
            '<answer>Paris</answer>',
            '<search>France capital</search>ignored suffix',
        ]

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {'input_ids': [10] * len(text)}


class ExactActionTokenizer(ActionTokenizer):
    def batch_decode(self, responses, skip_special_tokens=True):
        del responses, skip_special_tokens
        return [
            '<answer>Paris</answer>',
            '<search>France</search>discarded',
        ]

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            '</answer>': [8, 9],
            '</search>': [6, 7],
        }
        return {'input_ids': mapping.get(text, [10] * len(text))}


class UntaggedTokenizer(ExactActionTokenizer):
    def batch_decode(self, responses, skip_special_tokens=True):
        del responses, skip_special_tokens
        return ['unfinished reasoning']


class ObservationProcessingTest(unittest.TestCase):
    def setUp(self):
        self.manager = LLMGenerationManager.__new__(LLMGenerationManager)
        self.manager.tokenizer = CharacterTokenizer()
        self.manager.config = SimpleNamespace(max_obs_length=50)

    def test_truncates_each_observation_and_preserves_information_tags(self):
        long_observation = "\n\n<information>" + ("x" * 100) + "</information>\n\n"
        short_observation = "<information>short</information>"

        result, evidence_mask = self.manager._process_next_obs([long_observation, short_observation])
        decoded_long = self.manager.tokenizer.decode(result[0].tolist())
        decoded_short = self.manager.tokenizer.decode(result[1].tolist())

        self.assertEqual(result.shape[1], self.manager.config.max_obs_length)
        self.assertTrue(decoded_long.startswith("\n\n<information>"))
        self.assertTrue(decoded_long.endswith("</information>\n\n"))
        self.assertEqual(decoded_short, short_observation)
        self.assertEqual(evidence_mask[0].sum().item(), 19)
        self.assertEqual(evidence_mask[1].sum().item(), len("short"))

    def test_truncates_non_information_observation_normally(self):
        observation = "invalid action " * 10

        result, evidence_mask = self.manager._process_next_obs([observation])
        decoded = self.manager.tokenizer.decode(result[0].tolist())

        self.assertEqual(len(decoded), self.manager.config.max_obs_length)
        self.assertEqual(decoded, observation[:self.manager.config.max_obs_length])
        self.assertEqual(evidence_mask.sum().item(), 0)

    def test_final_answer_keeps_sampled_eos_but_search_action_drops_suffix_eos(self):
        manager = LLMGenerationManager.__new__(LLMGenerationManager)
        manager.tokenizer = ActionTokenizer()
        manager.config = SimpleNamespace(no_think_rl=False)
        raw_responses = torch.tensor([[1, 99, 0], [2, 99, 0]])

        processed, response_text = manager._postprocess_responses(raw_responses)

        self.assertEqual(response_text, ['<answer>Paris</answer>', '<search>France capital</search>'])
        self.assertEqual((processed[0] == manager.tokenizer.eos_token_id).sum().item(), 1)
        self.assertEqual((processed[1] == manager.tokenizer.eos_token_id).sum().item(), 0)

    def test_action_boundary_preserves_exact_sampled_ids_and_only_adjacent_answer_eos(self):
        manager = LLMGenerationManager.__new__(LLMGenerationManager)
        manager.tokenizer = ExactActionTokenizer()
        manager.config = SimpleNamespace(no_think_rl=False)
        raw_responses = torch.tensor([
            [21, 8, 9, 99, 0, 0],
            [31, 6, 7, 55, 99, 0],
        ])

        processed, _ = manager._postprocess_responses(raw_responses)

        self.assertEqual(processed[0, :4].tolist(), [21, 8, 9, 99])
        self.assertEqual(processed[1, :3].tolist(), [31, 6, 7])
        self.assertNotIn(55, processed[1].tolist())
        self.assertNotIn(99, processed[1].tolist())

    def test_untagged_response_preserves_exact_sampled_ids(self):
        manager = LLMGenerationManager.__new__(LLMGenerationManager)
        manager.tokenizer = UntaggedTokenizer()
        manager.config = SimpleNamespace(no_think_rl=False)

        processed, _ = manager._postprocess_responses(torch.tensor([[41, 42, 99, 0]]))

        self.assertEqual(processed[0].tolist(), [41, 42, 99])

    def test_allows_one_invalid_action_retry(self):
        next_obs, dones, valid_action, is_search, invalid_action = self.manager.execute_predictions(
            ["not a tagged action"],
            pad_token="",
            active_mask=[True],
            invalid_action_counts=[0],
        )

        self.assertIn("My previous action is invalid", next_obs[0])
        self.assertEqual(dones, [0])
        self.assertEqual(valid_action, [0])
        self.assertEqual(is_search, [0])
        self.assertEqual(invalid_action, [1])

    def test_second_invalid_action_terminates(self):
        next_obs, dones, valid_action, is_search, invalid_action = self.manager.execute_predictions(
            ["still not tagged"],
            pad_token="",
            active_mask=[True],
            invalid_action_counts=[1],
        )

        self.assertEqual(next_obs, [""])
        self.assertEqual(dones, [1])
        self.assertEqual(valid_action, [0])
        self.assertEqual(is_search, [0])
        self.assertEqual(invalid_action, [1])

    def test_final_rollout_invalid_action_terminates_without_retry(self):
        next_obs, dones, valid_action, is_search, invalid_action = self.manager.execute_predictions(
            ["not tagged on final rollout"],
            pad_token="",
            active_mask=[True],
            invalid_action_counts=[0],
            do_search=False,
        )

        self.assertEqual(next_obs, [""])
        self.assertEqual(dones, [1])
        self.assertEqual(valid_action, [0])
        self.assertEqual(is_search, [0])
        self.assertEqual(invalid_action, [1])


if __name__ == "__main__":
    unittest.main()
