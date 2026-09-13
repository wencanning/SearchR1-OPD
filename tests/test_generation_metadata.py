import unittest
from types import SimpleNamespace
import torch
from verl import DataProto
from search_r1.llm_agent.generation import LLMGenerationManager


class ReachedGeneration(Exception):
    pass


class GenerationMetadataTest(unittest.TestCase):
    def test_metadata_survives_regular_and_final_turn_selection(self):
        for turns in (0, 1):
            for validate in (False, True):
                with self.subTest(turns=turns, validate=validate):
                    metadata = {'validate': validate, 'recompute_log_prob': not validate,
                                'do_sample': False, 'eos_token_id': 99}
                    batch = DataProto.from_dict({'input_ids': torch.ones(3, 2, dtype=torch.long)})
                    batch.meta_info.update(metadata)
                    manager = LLMGenerationManager.__new__(LLMGenerationManager)
                    manager.config = SimpleNamespace(max_turns=turns, max_start_length=2)
                    manager.tensor_fn = SimpleNamespace(cut_to_effective_len=lambda batch, keys: batch)
                    def generate(selected):
                        self.assertEqual(selected.meta_info, metadata)
                        raise ReachedGeneration
                    manager._generate_with_gpu_padding = generate
                    with self.assertRaises(ReachedGeneration):
                        manager.run_llm_loop(batch, batch.batch['input_ids'])

    def test_metadata_survives_gpu_padding(self):
        for gpus in (1, 2, 3):
            manager = LLMGenerationManager.__new__(LLMGenerationManager)
            manager.config = SimpleNamespace(num_gpus=gpus)
            batch = DataProto.from_dict({'input_ids': torch.ones(3, 2, dtype=torch.long)})
            batch.meta_info.update(validate=True, recompute_log_prob=False, do_sample=False)
            def generate(selected):
                self.assertEqual(selected.meta_info, batch.meta_info)
                self.assertEqual(len(selected) % gpus, 0)
                raise ReachedGeneration
            manager.actor_rollout_wg = SimpleNamespace(generate_sequences=generate)
            with self.assertRaises(ReachedGeneration):
                manager._generate_with_gpu_padding(batch)


if __name__ == '__main__':
    unittest.main()
