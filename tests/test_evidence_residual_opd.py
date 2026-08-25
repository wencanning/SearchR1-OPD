import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import validate_opd_teacher_target_config
from verl.workers.actor.dp_actor import (
    DataParallelPPOActor,
    _trim_shared_prompt_padding,
)
from verl.utils.tokenizer import (
    mask_logits_to_tokenizer_vocab,
    mask_vllm_logits_to_tokenizer_vocab,
    validate_same_model_vocab,
    validate_same_tokenizer_vocab,
)


class _FakeCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=7, hidden_size=5):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        use_cache,
        num_logits_to_keep,
    ):
        del attention_mask, position_ids, use_cache
        hidden_states = self.embedding(input_ids)[:, -num_logits_to_keep:]
        return SimpleNamespace(logits=self.lm_head(hidden_states))
from verl.utils.evidence_residual import (
    build_evidence_hidden_attention_mask,
    calibrate_entropy_matched_temperature,
    compute_evidence_residual_target_stats,
    compute_temperature_target_stats,
    select_calibration_ids,
)


class EvidenceResidualTargetTest(unittest.TestCase):
    def test_teacher_lm_head_projects_only_policy_rows_without_changing_logits(self):
        torch.manual_seed(3)
        model = _FakeCausalLM()
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.config = OmegaConf.create({
            'target_forward_dtype': 'fp32',
            'target_select_policy_logits': True,
            'allow_tf32': False,
        })
        actor.actor_module = model
        actor.logit_vocab_size = 7
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.size(-1)).unsqueeze(0)
        target_row_mask = torch.tensor([[True, False, True]])

        with torch.no_grad():
            full_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                num_logits_to_keep=4,
            ).logits[:, :-1][target_row_mask]
            selected_logits = actor._forward_padded_teacher_logits(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                response_length=3,
                target_row_mask=target_row_mask,
            )

        self.assertEqual(selected_logits.shape, (2, 7))
        torch.testing.assert_close(selected_logits, full_logits)

    def test_teacher_trim_removes_only_shared_left_prompt_padding(self):
        input_ids = torch.arange(16).reshape(2, 8)
        attention_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
        ])
        position_ids = torch.tensor([
            [0, 0, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 1, 2, 3, 4],
        ])
        evidence_mask = torch.zeros_like(input_ids)
        evidence_mask[:, 5] = 1

        trimmed = _trim_shared_prompt_padding(
            input_ids,
            attention_mask,
            position_ids,
            evidence_mask,
            response_length=3,
        )

        for actual, expected in zip(
            trimmed,
            (input_ids[:, 2:], attention_mask[:, 2:], position_ids[:, 2:], evidence_mask[:, 2:]),
        ):
            torch.testing.assert_close(actual, expected)

    def test_hidden_mask_preserves_slots_and_blocks_only_evidence_key_columns(self):
        attention_mask = torch.tensor([[0, 1, 1, 1]])
        evidence_mask = torch.tensor([[0, 0, 1, 0]])
        result = build_evidence_hidden_attention_mask(
            attention_mask,
            evidence_mask,
            torch.float32,
        )
        blocked = torch.finfo(torch.float32).min

        self.assertEqual(result.shape, (1, 1, 4, 4))
        self.assertEqual(result[0, 0, 3, 2].item(), blocked)
        self.assertEqual(result[0, 0, 3, 1].item(), 0.0)
        self.assertEqual(result[0, 0, 3, 3].item(), 0.0)
        self.assertEqual(result[0, 0, 1, 0].item(), blocked)
        self.assertEqual(result[0, 0, 1, 3].item(), blocked)

    def test_hidden_mask_rejects_marked_padding(self):
        with self.assertRaisesRegex(ValueError, 'padding'):
            build_evidence_hidden_attention_mask(
                torch.tensor([[0, 1]]),
                torch.tensor([[1, 0]]),
                torch.float32,
            )

    def test_er_target_matches_full_vocab_formula(self):
        observed = torch.tensor([[[1.0, 0.0, -1.0], [0.2, -0.5, 1.2]]])
        hidden = torch.tensor([[[0.0, 0.5, -0.5], [-0.1, 0.3, 0.7]]])
        labels = torch.tensor([[0, 2]])

        stats = compute_evidence_residual_target_stats(
            observed,
            hidden,
            labels,
            token_chunk_size=1,
        )
        expected_logits = 2.0 * observed - hidden
        expected_log_probs = torch.log_softmax(expected_logits, dim=-1)
        expected_selected = expected_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        expected_entropy = -(expected_log_probs.exp() * expected_log_probs).sum(dim=-1)

        torch.testing.assert_close(stats['target_log_prob'], expected_selected)
        torch.testing.assert_close(stats['target_entropy'], expected_entropy)
        torch.testing.assert_close(
            stats['selected_logit_delta'],
            (observed - hidden).gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        )

    def test_er_target_is_finite_for_large_vocab_and_wide_logits(self):
        vocab_size = 152064
        observed = torch.linspace(-80.0, 80.0, vocab_size).reshape(1, 1, -1)
        hidden = torch.linspace(60.0, -60.0, vocab_size).reshape(1, 1, -1)
        labels = torch.tensor([[vocab_size - 1]])

        stats = compute_evidence_residual_target_stats(
            observed,
            hidden,
            labels,
            token_chunk_size=1,
        )
        expected_log_probs = torch.log_softmax(2.0 * observed - hidden, dim=-1)
        expected_selected = expected_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        expected_entropy = -(expected_log_probs.exp() * expected_log_probs).sum(dim=-1)

        self.assertTrue(all(torch.isfinite(value).all() for value in stats.values()))
        torch.testing.assert_close(stats['target_log_prob'], expected_selected)
        torch.testing.assert_close(stats['target_entropy'], expected_entropy)

    def test_er_target_rejects_non_finite_teacher_logits(self):
        observed = torch.tensor([[[0.0, float('nan')]]])
        hidden = torch.zeros_like(observed)

        with self.assertRaisesRegex(FloatingPointError, 'non-finite'):
            compute_evidence_residual_target_stats(
                observed,
                hidden,
                torch.tensor([[0]]),
                token_chunk_size=1,
            )

    def test_temperature_control_uses_one_global_tau(self):
        logits = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        labels = torch.tensor([[0, 1]])
        stats = compute_temperature_target_stats(logits, labels, temperature=2.0, token_chunk_size=1)
        expected = torch.log_softmax(logits / 2.0, dim=-1)
        torch.testing.assert_close(
            stats['target_log_prob'],
            expected.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        )

    def test_entropy_calibration_recovers_global_temperature(self):
        logits = torch.tensor([
            [2.0, 0.5, -1.0],
            [0.1, 1.3, -0.2],
            [3.0, 2.0, 0.0],
        ])

        def mean_entropy(temperature):
            log_probs = torch.log_softmax(logits / temperature, dim=-1)
            return float((-(log_probs.exp() * log_probs).sum(dim=-1)).mean())

        expected_tau = 0.7
        result = calibrate_entropy_matched_temperature(
            mean_entropy,
            target_entropy=mean_entropy(expected_tau),
        )
        self.assertAlmostEqual(result.temperature, expected_tau, places=5)
        self.assertLessEqual(result.entropy_gap, 1e-6)

    def test_calibration_split_is_seeded_and_order_independent(self):
        ids = [f'q-{idx}' for idx in range(100)]
        selected = select_calibration_ids(ids, seed=17)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, select_calibration_ids(reversed(ids), seed=17))
        self.assertNotEqual(selected, select_calibration_ids(ids, seed=18))


class TeacherTargetConfigTest(unittest.TestCase):
    @staticmethod
    def _config(target='evidence_residual'):
        return OmegaConf.create({
            'do_search': True,
            'data': {'train_batch_size': 8},
            'algorithm': {
                'opd': {
                    'teacher_target': target,
                    'advantage_mode': 'token',
                    'normalize': False,
                    'clip_value': None,
                    'grpo_reward_coef': 0.0,
                    'lambda_distill': 1.0,
                    'target_token_chunk_size': 16,
                    'entropy_matched_tau': 0.7 if target == 'entropy_matched' else None,
                },
            },
            'actor_rollout_ref': {
                'actor': {
                    'state_masking': True,
                    'entropy_coeff': 0.0,
                    'ppo_epochs': 1,
                    'ppo_mini_batch_size': 8,
                },
                'rollout': {
                    'temperature': 1.0,
                    'restrict_to_tokenizer_vocab': True,
                    'n': 1,
                    'n_agent': 1,
                },
                'ref': {
                    'attn_implementation': 'sdpa',
                    'ulysses_sequence_parallel_size': 1,
                    'target_forward_dtype': 'fp32',
                    'allow_approximate_target_precision': False,
                    'fsdp_config': {
                        'model_dtype': 'fp32',
                        'mixed_precision': {'param_dtype': 'fp32'},
                    },
                },
            },
        })

    def test_er_config_accepts_pure_token_opd(self):
        validate_opd_teacher_target_config(self._config())

    def test_er_config_accepts_grpo_with_group_rollouts(self):
        config = self._config()
        config.algorithm.opd.grpo_reward_coef = 1.0
        config.actor_rollout_ref.rollout.n_agent = 8
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 16
        validate_opd_teacher_target_config(config)

    def test_er_config_rejects_grpo_without_group_rollouts(self):
        config = self._config()
        config.algorithm.opd.grpo_reward_coef = 1.0
        with self.assertRaisesRegex(ValueError, 'more than one rollout'):
            validate_opd_teacher_target_config(config)

    def test_pure_er_config_rejects_multiple_rollouts(self):
        config = self._config()
        config.actor_rollout_ref.rollout.n_agent = 8
        with self.assertRaisesRegex(ValueError, 'pure evidence_residual'):
            validate_opd_teacher_target_config(config)

    def test_entropy_control_requires_calibrated_tau(self):
        config = self._config('entropy_matched')
        config.algorithm.opd.entropy_matched_tau = None
        with self.assertRaisesRegex(ValueError, 'entropy_matched_tau'):
            validate_opd_teacher_target_config(config)

    def test_er_config_rejects_bf16_teacher_logits(self):
        config = self._config()
        config.actor_rollout_ref.ref.target_forward_dtype = 'bfloat16'
        with self.assertRaisesRegex(ValueError, 'target_forward_dtype=fp32'):
            validate_opd_teacher_target_config(config)

    def test_er_config_accepts_explicit_fp16_teacher(self):
        config = self._config()
        config.actor_rollout_ref.ref.allow_approximate_target_precision = True
        config.actor_rollout_ref.ref.target_forward_dtype = 'float16'
        config.actor_rollout_ref.ref.fsdp_config.model_dtype = 'float16'
        config.actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype = 'float16'
        validate_opd_teacher_target_config(config)

    def test_er_config_rejects_mixed_approximate_precision(self):
        config = self._config()
        config.actor_rollout_ref.ref.allow_approximate_target_precision = True
        config.actor_rollout_ref.ref.target_forward_dtype = 'float16'
        with self.assertRaisesRegex(ValueError, 'consistently use FP16 or BF16'):
            validate_opd_teacher_target_config(config)


class _FakeTokenizer:
    def __init__(self, tokens, added_items, special_ids):
        self.tokens = tokens
        self._added = dict(added_items)
        self.all_special_ids = special_ids
        self.special_tokens_map = {'eos_token': tokens[special_ids[0]]}

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_ids):
        return [self.tokens[token_id] for token_id in token_ids]

    def get_added_vocab(self):
        return self._added


class TokenizerAlignmentTest(unittest.TestCase):
    def test_rollout_vocab_mask_keeps_added_tokens_and_blocks_head_padding(self):
        logits = torch.zeros(6)
        result = mask_logits_to_tokenizer_vocab([], logits, vocab_size=4)
        self.assertTrue(torch.equal(result[:4], torch.zeros(4)))
        self.assertTrue(torch.isneginf(result[4:]).all())

    def test_rollout_vocab_mask_rejects_short_model_head(self):
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            mask_logits_to_tokenizer_vocab([], torch.zeros(3), vocab_size=4)

    def test_vllm_vocab_mask_adapter_has_three_argument_signature(self):
        import inspect
        from functools import partial

        processor = partial(mask_vllm_logits_to_tokenizer_vocab, 4)
        self.assertEqual(len(inspect.signature(processor).parameters), 3)
        logits = torch.zeros(6)
        result = processor([1, 2], [3], logits)
        self.assertTrue(torch.equal(result[:4], torch.zeros(4)))
        self.assertTrue(torch.isneginf(result[4:]).all())

    def test_exact_coordinate_alignment_passes(self):
        student = _FakeTokenizer(['a', 'b', '<eos>'], [('<eos>', 2)], [2])
        teacher = _FakeTokenizer(['a', 'b', '<eos>'], [('<eos>', 2)], [2])
        validate_same_tokenizer_vocab(student, teacher)

    def test_permuted_coordinates_fail(self):
        student = _FakeTokenizer(['a', 'b', '<eos>'], [('<eos>', 2)], [2])
        teacher = _FakeTokenizer(['b', 'a', '<eos>'], [('<eos>', 2)], [2])
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            validate_same_tokenizer_vocab(student, teacher)

    def test_each_model_head_must_cover_shared_tokenizer_vocab(self):
        tokenizer = _FakeTokenizer(['a', 'b', '<eos>'], [('<eos>', 2)], [2])
        validate_same_model_vocab(
            OmegaConf.create({'vocab_size': 4}),
            OmegaConf.create({'vocab_size': 5}),
            tokenizer,
        )
        with self.assertRaisesRegex(ValueError, 'teacher model output'):
            validate_same_model_vocab(
                OmegaConf.create({'vocab_size': 4}),
                OmegaConf.create({'vocab_size': 2}),
                tokenizer,
            )


if __name__ == '__main__':
    unittest.main()
