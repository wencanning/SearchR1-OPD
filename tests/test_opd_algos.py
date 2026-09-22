import unittest

import numpy as np
import torch

from verl import DataProto
from search_r1.diagnostics.opd_uncertainty import infer_evidence_step_ids, retrieval_hits_by_information_block
from verl.trainer.ppo.core_algos import (
    compute_opd_advantage,
    compute_policy_loss,
    compute_rce_opd_advantage,
    compute_sod_stepwise_weights,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _build_rce_retrieval_hit_values,
    compute_advantage,
    compute_data_metrics,
    compute_opd_logprob_metrics,
)
from verl.utils.torch_functional import masked_mean


class TestOPDAdvantage(unittest.TestCase):
    def test_token_advantage_uses_teacher_student_log_ratio(self):
        student = torch.tensor([[-2.0, -1.0, -4.0]])
        teacher = torch.tensor([[-1.0, -2.0, -1.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        advantages, returns = compute_opd_advantage(student, teacher, mask)

        expected = torch.tensor([[1.0, -1.0, 0.0]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(returns, expected)

    def test_sequence_advantage_is_future_cumulative_signal(self):
        student = torch.tensor([[-2.0, -1.0, -4.0]])
        teacher = torch.tensor([[-1.0, -2.0, -1.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        advantages, _ = compute_opd_advantage(
            student,
            teacher,
            mask,
            advantage_mode='sequence',
        )

        torch.testing.assert_close(advantages, torch.tensor([[0.0, -1.0, 0.0]]))

    def test_advantage_can_be_clipped(self):
        student = torch.tensor([[-5.0, -1.0]])
        teacher = torch.tensor([[-1.0, -5.0]])
        mask = torch.ones_like(student)

        advantages, _ = compute_opd_advantage(
            student,
            teacher,
            mask,
            clip_value=0.5,
        )

        torch.testing.assert_close(advantages, torch.tensor([[0.5, -0.5]]))

    def test_observation_tokens_are_excluded_from_advantage(self):
        student = torch.tensor([[-2.0, -20.0, -3.0]])
        teacher = torch.tensor([[-1.0, 20.0, -1.0]])
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])

        token_advantages, _ = compute_opd_advantage(student, teacher, loss_mask)
        sequence_advantages, _ = compute_opd_advantage(
            student,
            teacher,
            loss_mask,
            advantage_mode='sequence',
        )

        torch.testing.assert_close(token_advantages, torch.tensor([[1.0, 0.0, 2.0]]))
        torch.testing.assert_close(sequence_advantages, torch.tensor([[3.0, 0.0, 2.0]]))

    def test_trainer_uses_loss_mask_for_opd_advantage_and_metrics(self):
        batch = DataProto.from_dict(tensors={
            'responses': torch.ones(1, 3, dtype=torch.long),
            'attention_mask': torch.ones(1, 5, dtype=torch.long),
            'loss_mask': torch.tensor([[1, 0, 1]], dtype=torch.long),
            'old_log_probs': torch.tensor([[-2.0, -20.0, -3.0]]),
            'ref_log_prob': torch.tensor([[-1.0, 20.0, -1.0]]),
            'token_level_scores': torch.tensor([[1.0, 0.0, 2.0]]),
            'token_level_rewards': torch.tensor([[1.0, 0.0, 2.0]]),
        })
        batch.meta_info['opd_config'] = {
            'advantage_mode': 'token',
            'normalize': False,
            'clip_value': None,
        }
        batch.meta_info['sampled_token_preservation_rate'] = 1.0
        batch.meta_info['canonical_retokenization_mismatch_rate'] = 0.25

        output = compute_advantage(batch, 'opd')
        metrics = compute_data_metrics(output, use_critic=False)

        torch.testing.assert_close(output.batch['advantages'], torch.tensor([[1.0, 0.0, 2.0]]))
        self.assertNotIn('opd_sod_stepwise_weights', output.batch)
        self.assertNotIn('opd_sod_step_divergence', output.batch)
        self.assertEqual(metrics['critic/advantages/mean'], 1.5)
        self.assertEqual(metrics['critic/advantages/min'], 1.0)
        self.assertEqual(metrics['rollout/sampled_token_preservation_rate'], 1.0)
        self.assertEqual(
            metrics['rollout/canonical_retokenization_sequence_mismatch_rate'],
            0.25,
        )

    def test_protocol_tags_are_excluded_only_from_opd_not_grpo(self):
        batch = DataProto.from_dict(
            tensors={
                'responses': torch.ones(2, 3, dtype=torch.long),
                'attention_mask': torch.ones(2, 5, dtype=torch.long),
                'loss_mask': torch.ones(2, 3, dtype=torch.long),
                'opd_distillation_mask': torch.tensor(
                    [[0, 1, 0], [0, 1, 0]], dtype=torch.long
                ),
                'old_log_probs': torch.zeros(2, 3),
                'ref_log_prob': torch.ones(2, 3),
                'token_level_rewards': torch.tensor(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
                ),
            },
            non_tensors={'uid': np.array(['prompt', 'prompt'], dtype=object)},
        )
        batch.meta_info['opd_config'] = {
            'advantage_mode': 'token',
            'normalize': False,
            'clip_value': None,
            'distillation_coef': 1.0,
            'grpo_reward_coef': 1.0,
        }

        output = compute_advantage(batch, 'opd')

        torch.testing.assert_close(
            output.batch['opd_advantages'],
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        self.assertTrue(torch.all(output.batch['weighted_opd_advantages'][:, [0, 2]] == 0))
        self.assertTrue(torch.all(output.batch['grpo_advantages'][:, [0, 2]] != 0))
        torch.testing.assert_close(
            output.batch['advantages'][:, [0, 2]],
            output.batch['grpo_advantages'][:, [0, 2]],
        )

    def test_protocol_tag_mask_tracks_split_tag_tokens(self):
        class CharacterTokenizer:
            def decode(
                self,
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ):
                del skip_special_tokens, clean_up_tokenization_spaces
                return ''.join(chr(token_id) for token_id in token_ids)

        text = '<think>x</think><search>q</search><answer>a</answer>'
        response_ids = torch.tensor([[ord(char) for char in text]], dtype=torch.long)
        batch = DataProto.from_dict(tensors={
            'responses': response_ids,
            'attention_mask': torch.ones(1, response_ids.shape[1] + 2, dtype=torch.long),
            'loss_mask': torch.ones_like(response_ids),
        })
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.tokenizer = CharacterTokenizer()

        output, metrics = trainer._create_opd_distillation_mask(
            batch, {}, mask_protocol_tags=True
        )

        kept_text = ''.join(
            char
            for char, keep in zip(
                text,
                output.batch['opd_distillation_mask'][0].tolist(),
            )
            if keep
        )
        self.assertEqual(kept_text, 'xqa')
        self.assertTrue(torch.all(output.batch['loss_mask'] == 1))
        self.assertEqual(metrics['opd/config/mask_protocol_tags'], 1.0)
        self.assertGreater(metrics['opd/protocol_tag_token_fraction'], 0.0)

        output, metrics = trainer._create_opd_distillation_mask(batch, {})
        self.assertTrue(torch.all(output.batch['opd_distillation_mask'] == 1))
        self.assertEqual(metrics['opd/config/mask_protocol_tags'], 0.0)

    def test_opd_divergence_matches_sod_absolute_logprob_gap(self):
        old_log_probs = torch.tensor([[-2.0, -20.0, -3.0]])
        ref_log_prob = torch.tensor([[-1.0, 20.0, -1.0]])
        loss_mask = torch.tensor([[1, 0, 1]], dtype=torch.long)

        metrics = compute_opd_logprob_metrics(old_log_probs, ref_log_prob, loss_mask)

        self.assertEqual(metrics['opd/divergence'], 1.5)
        self.assertEqual(metrics['opd/reverse_kl_k1'], -1.5)
        self.assertEqual(metrics['opd/teacher_advantage'], 1.5)

    def test_sod_weights_contiguous_assistant_steps(self):
        student = torch.zeros(1, 8)
        teacher = torch.tensor([[1.0, 1.0, 99.0, 99.0, 2.0, 2.0, 99.0, 0.5]])
        action_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 1]])

        weights, divergence = compute_sod_stepwise_weights(
            student,
            teacher,
            action_mask,
            epsilon=1e-6,
            delta=0.2,
        )

        second_weight = (1.0 + 1e-6) / (2.0 + 1e-6)
        expected_weights = torch.tensor([[
            1.0, 1.0, 0.0, 0.0,
            second_weight, second_weight, 0.0, 1.2,
        ]])
        expected_divergence = torch.tensor([[
            1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.5,
        ]])

        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(divergence, expected_divergence)

    def test_single_rollout_sod_without_grpo_is_reward_independent(self):
        mask = torch.tensor([[1, 1, 0, 1, 1, 0]], dtype=torch.long)
        batch = DataProto.from_dict(tensors={
            'responses': torch.ones(1, 6, dtype=torch.long),
            'attention_mask': torch.ones(1, 8, dtype=torch.long),
            'loss_mask': mask,
            'old_log_probs': torch.full((1, 6), -3.0),
            'ref_log_prob': torch.tensor([[-2., -2., -1., -1., -1., -1.]]),
            'token_level_rewards': torch.zeros(1, 6),
        }, non_tensors={'uid': np.array(['single_prompt'], dtype=object)})
        batch.meta_info['opd_config'] = {
            'advantage_mode': 'token', 'normalize': False, 'clip_value': None,
            'distillation_coef': .01, 'grpo_reward_coef': 0.,
            'use_gated_distillation': False,
            'sod': {'enable': True, 'epsilon': 1e-6, 'delta': .2},
        }
        first = compute_advantage(batch, 'opd').batch['advantages'].clone()
        batch.batch['token_level_rewards'].fill_(100.)
        second = compute_advantage(batch, 'opd').batch['advantages']
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(second).all())
        self.assertGreater(second.abs().sum().item(), 0)
        torch.testing.assert_close(second * (1-mask), torch.zeros_like(second))

    def test_sod_combines_stepwise_opd_with_grpo(self):
        action_mask = torch.tensor([
            [1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
        ], dtype=torch.long)
        teacher_log_prob = torch.tensor([
            [1.0, 1.0, 50.0, 2.0, 2.0, 50.0],
            [1.0, 1.0, 50.0, 2.0, 2.0, 50.0],
        ])
        batch = DataProto.from_dict(
            tensors={
                'responses': torch.ones(2, 6, dtype=torch.long),
                'attention_mask': torch.ones(2, 8, dtype=torch.long),
                'loss_mask': action_mask,
                'old_log_probs': torch.zeros(2, 6),
                'ref_log_prob': teacher_log_prob,
                'token_level_rewards': torch.tensor([
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]),
            },
            non_tensors={'uid': np.array(['prompt', 'prompt'], dtype=object)},
        )
        batch.meta_info['opd_config'] = {
            'advantage_mode': 'token',
            'normalize': False,
            'clip_value': None,
            'distillation_coef': 1.0,
            'grpo_reward_coef': 1.0,
            'use_gated_distillation': False,
            'sod': {
                'enable': True,
                'epsilon': 1e-6,
                'delta': 0.2,
            },
        }

        output = compute_advantage(batch, 'opd')

        second_weight = (1.0 + 1e-6) / (2.0 + 1e-6)
        expected_weights = torch.tensor([
            [1.0, 1.0, 0.0, second_weight, second_weight, 0.0],
            [1.0, 1.0, 0.0, second_weight, second_weight, 0.0],
        ])
        expected_weighted_opd = action_mask.float()

        torch.testing.assert_close(
            output.batch['opd_sod_stepwise_weights'], expected_weights
        )
        torch.testing.assert_close(
            output.batch['opd_effective_distillation_coef'], expected_weights
        )
        torch.testing.assert_close(
            output.batch['weighted_opd_advantages'], expected_weighted_opd
        )
        torch.testing.assert_close(
            output.batch['advantages'],
            output.batch['grpo_advantages'] + expected_weighted_opd,
        )

    def test_opd_can_add_grpo_outcome_advantage(self):
        batch = DataProto.from_dict(
            tensors={
                'responses': torch.ones(2, 3, dtype=torch.long),
                'attention_mask': torch.ones(2, 5, dtype=torch.long),
                'loss_mask': torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.long),
                'old_log_probs': torch.zeros(2, 3),
                'ref_log_prob': torch.zeros(2, 3),
                'token_level_rewards': torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
            },
            non_tensors={'uid': np.array(['prompt', 'prompt'], dtype=object)},
        )
        batch.meta_info['opd_config'] = {
            'advantage_mode': 'token',
            'normalize': False,
            'clip_value': None,
            'distillation_coef': 1.0,
            'grpo_reward_coef': 1.0,
        }

        output = compute_advantage(batch, 'opd')

        torch.testing.assert_close(output.batch['opd_advantages'], torch.zeros(2, 3))
        torch.testing.assert_close(output.batch['advantages'], output.batch['grpo_advantages'])
        self.assertTrue(torch.all(output.batch['advantages'][:, 1] == 0))

        self.assertGreater(output.batch['advantages'][0, 0].item(), 0)
        self.assertLess(output.batch['advantages'][1, 0].item(), 0)

    def test_opd_rl_uses_sigmoid_beta_gate_for_distillation(self):
        batch = DataProto.from_dict(
            tensors={
                'responses': torch.ones(2, 2, dtype=torch.long),
                'attention_mask': torch.ones(2, 4, dtype=torch.long),
                'loss_mask': torch.ones(2, 2, dtype=torch.long),
                'old_log_probs': torch.zeros(2, 2),
                'ref_log_prob': torch.ones(2, 2),
                'token_level_rewards': torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            },
            non_tensors={'uid': np.array(['prompt', 'prompt'], dtype=object)},
        )
        batch.meta_info['opd_config'] = {
            'advantage_mode': 'token',
            'normalize': False,
            'clip_value': None,
            'distillation_coef': 1.0,
            'grpo_reward_coef': 1.0,
            'use_gated_distillation': True,
            'gamma': 1.0,
            'beta_min': 0.0,
            'beta_max': 0.05,
        }

        output = compute_advantage(batch, 'opd')

        mask = torch.ones(2, 2)
        task_advantages = masked_mean(output.batch['grpo_advantages'], mask, axis=1)
        expected_beta = 0.05 * torch.sigmoid(-task_advantages)
        expected_beta = expected_beta.unsqueeze(-1).expand_as(output.batch['opd_beta'])

        torch.testing.assert_close(output.batch['opd_advantages'], torch.ones(2, 2))
        torch.testing.assert_close(output.batch['opd_beta'], expected_beta)
        torch.testing.assert_close(output.batch['opd_effective_distillation_coef'], expected_beta)
        torch.testing.assert_close(output.batch['weighted_opd_advantages'], expected_beta)
        torch.testing.assert_close(
            output.batch['advantages'],
            output.batch['grpo_advantages'] + expected_beta,
        )

    def test_rce_opd_weights_steps_by_retrieval_and_teacher_entropy(self):
        student = torch.zeros(1, 4)
        teacher = torch.ones(1, 4)
        teacher_entropy = torch.tensor([[0.2, 0.4, 2.0, 2.2]])
        mask = torch.ones_like(student)
        retrieval_hit = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        step_ids = torch.tensor([[0, 0, 1, 1]])

        advantages, _, weights = compute_rce_opd_advantage(
            old_log_prob=student,
            teacher_log_prob=teacher,
            teacher_entropy=teacher_entropy,
            eos_mask=mask,
            retrieval_hit=retrieval_hit,
            step_ids=step_ids,
            w_min=0.1,
            w_max=1.0,
            alpha=4.0,
            tau=0.0,
            return_weights=True,
        )

        high_weight = 0.1 + 0.9 * torch.sigmoid(torch.tensor(4.0))
        low_weight = 0.1 + 0.9 * torch.sigmoid(torch.tensor(-4.0))
        expected = torch.tensor([[
            high_weight.item(),
            high_weight.item(),
            low_weight.item(),
            low_weight.item(),
        ]])

        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(advantages, expected)

    def test_trainer_can_use_rce_opd_as_separate_advantage_interface(self):
        batch = DataProto.from_dict(
            tensors={
                'responses': torch.ones(1, 4, dtype=torch.long),
                'attention_mask': torch.ones(1, 6, dtype=torch.long),
                'loss_mask': torch.ones(1, 4, dtype=torch.long),
                'old_log_probs': torch.zeros(1, 4),
                'ref_log_prob': torch.ones(1, 4),
                'ref_entropy': torch.tensor([[0.2, 0.4, 2.0, 2.2]]),
                'rce_retrieval_hit': torch.tensor([1.0]),
                'rce_step_ids': torch.tensor([[0, 0, 1, 1]]),
                'token_level_rewards': torch.zeros(1, 4),
            },
            non_tensors={'uid': np.array(['prompt'], dtype=object)},
        )
        batch.meta_info['opd_config'] = {
            'advantage_mode': 'token',
            'normalize': False,
            'clip_value': None,
            'distillation_coef': 2.0,
            'grpo_reward_coef': 0.0,
            'rce': {
                'enable': True,
                'entropy_normalization': 'percentile_rank',
                'w_min': 0.1,
                'w_max': 1.0,
                'alpha': 4.0,
                'tau': 0.0,
                'default_retrieval_hit': 0.5,
            },
        }

        output = compute_advantage(batch, 'opd')

        expected_step0 = 0.1 + 0.9 * torch.sigmoid(torch.tensor(4.0))
        expected_step1 = 0.1 + 0.9 * torch.sigmoid(torch.tensor(0.0))
        expected = torch.tensor([[
            expected_step0.item(),
            expected_step0.item(),
            expected_step1.item(),
            expected_step1.item(),
        ]])

        torch.testing.assert_close(output.batch['opd_rce_weights'], expected)
        torch.testing.assert_close(output.batch['opd_advantages'], expected)
        torch.testing.assert_close(output.batch['opd_beta'], torch.ones_like(expected))
        torch.testing.assert_close(output.batch['opd_effective_distillation_coef'], torch.full_like(expected, 2.0))
        torch.testing.assert_close(output.batch['advantages'], 2.0 * expected)

    def test_rce_metadata_helpers_track_previous_information_step(self):
        token_texts = [
            "<think>ask</think><search>q1</search>",
            "<information>miss</information>",
            "<think>use doc 1</think><search>q2</search>",
            "<information>gold answer</information>",
            "<think>use doc 2</think><answer>gold answer</answer>",
        ]

        step_ids = infer_evidence_step_ids(token_texts)
        hits = retrieval_hits_by_information_block("".join(token_texts), ["gold answer"])

        self.assertEqual(step_ids, [0, 0, 1, 1, 2])
        self.assertEqual(hits, [False, True])

    def test_rce_retrieval_hit_defaults_only_before_first_retrieval(self):
        values = _build_rce_retrieval_hit_values(
            evidence_step_ids=[0, 0, 1, 1, 2, 3],
            block_hits=[False, True],
            pre_retrieval_hit=0.5,
        )

        self.assertEqual(values, [0.5, 0.5, 0.0, 0.0, 1.0, 0.0])


class TestDualClipPPO(unittest.TestCase):
    def test_dual_clip_caps_large_negative_advantage_loss(self):
        old_log_prob = torch.zeros(1, 1)
        log_prob = torch.log(torch.tensor([[5.0]]))
        advantages = torch.tensor([[-1.0]])
        mask = torch.ones(1, 1)

        pg_loss, _, _, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob,
            log_prob,
            advantages,
            mask,
            cliprange=0.2,
            clip_ratio_c=3.0,
        )

        torch.testing.assert_close(pg_loss, torch.tensor(3.0))
        torch.testing.assert_close(pg_clipfrac_lower, torch.tensor(1.0))

    def test_positive_advantage_uses_asymmetric_upper_clip(self):
        old_log_prob = torch.zeros(1, 1)
        log_prob = torch.log(torch.tensor([[5.0]]))
        advantages = torch.tensor([[1.0]])
        mask = torch.ones(1, 1)

        pg_loss, _, _, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob,
            log_prob,
            advantages,
            mask,
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.28,
            clip_ratio_c=3.0,
        )

        torch.testing.assert_close(pg_loss, torch.tensor(-1.28))
        torch.testing.assert_close(pg_clipfrac_lower, torch.tensor(0.0))

    def test_policy_loss_excludes_observation_tokens(self):
        old_log_prob = torch.zeros(1, 2)
        log_prob = torch.log(torch.tensor([[5.0, 100.0]]))
        advantages = torch.tensor([[-1.0, -100.0]])
        loss_mask = torch.tensor([[1.0, 0.0]])

        pg_loss, _, _, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob,
            log_prob,
            advantages,
            loss_mask,
            cliprange=0.2,
            clip_ratio_c=3.0,
        )

        torch.testing.assert_close(pg_loss, torch.tensor(3.0))
        torch.testing.assert_close(pg_clipfrac_lower, torch.tensor(1.0))


if __name__ == '__main__':
    unittest.main()
