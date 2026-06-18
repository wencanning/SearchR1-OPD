import unittest

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_opd_advantage, compute_policy_loss
from verl.trainer.ppo.ray_trainer import compute_advantage, compute_data_metrics, compute_opd_logprob_metrics
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

        output = compute_advantage(batch, 'opd')
        metrics = compute_data_metrics(output, use_critic=False)

        torch.testing.assert_close(output.batch['advantages'], torch.tensor([[1.0, 0.0, 2.0]]))
        self.assertEqual(metrics['critic/advantages/mean'], 1.5)
        self.assertEqual(metrics['critic/advantages/min'], 1.0)

    def test_opd_divergence_matches_sod_absolute_logprob_gap(self):
        old_log_probs = torch.tensor([[-2.0, -20.0, -3.0]])
        ref_log_prob = torch.tensor([[-1.0, 20.0, -1.0]])
        loss_mask = torch.tensor([[1, 0, 1]], dtype=torch.long)

        metrics = compute_opd_logprob_metrics(old_log_probs, ref_log_prob, loss_mask)

        self.assertEqual(metrics['opd/divergence'], 1.5)
        self.assertEqual(metrics['opd/reverse_kl_k1'], -1.5)
        self.assertEqual(metrics['opd/teacher_advantage'], 1.5)

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
        torch.testing.assert_close(output.batch['weighted_opd_advantages'], expected_beta)
        torch.testing.assert_close(
            output.batch['advantages'],
            output.batch['grpo_advantages'] + expected_beta,
        )


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
