import unittest
from types import SimpleNamespace

from search_r1.llm_agent.generation import LLMGenerationManager


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids if token_id != self.pad_token_id)


class ObservationProcessingTest(unittest.TestCase):
    def setUp(self):
        self.manager = LLMGenerationManager.__new__(LLMGenerationManager)
        self.manager.tokenizer = CharacterTokenizer()
        self.manager.config = SimpleNamespace(max_obs_length=50)

    def test_truncates_each_observation_and_preserves_information_tags(self):
        long_observation = "\n\n<information>" + ("x" * 100) + "</information>\n\n"
        short_observation = "<information>short</information>"

        result = self.manager._process_next_obs([long_observation, short_observation])
        decoded_long = self.manager.tokenizer.decode(result[0].tolist())
        decoded_short = self.manager.tokenizer.decode(result[1].tolist())

        self.assertEqual(result.shape[1], self.manager.config.max_obs_length)
        self.assertTrue(decoded_long.startswith("\n\n<information>"))
        self.assertTrue(decoded_long.endswith("</information>\n\n"))
        self.assertEqual(decoded_short, short_observation)

    def test_truncates_non_information_observation_normally(self):
        observation = "invalid action " * 10

        result = self.manager._process_next_obs([observation])
        decoded = self.manager.tokenizer.decode(result[0].tolist())

        self.assertEqual(len(decoded), self.manager.config.max_obs_length)
        self.assertEqual(decoded, observation[:self.manager.config.max_obs_length])


if __name__ == "__main__":
    unittest.main()
