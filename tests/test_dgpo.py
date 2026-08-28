import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    apply_kl_penalty,
    compute_advantage,
    create_dgpo_reward_mask,
    validate_dgpo_config,
)


class DGPOSelectiveKLTest(unittest.TestCase):
    def test_official_reward_mask_guides_only_incorrect_trajectories(self):
        scores = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.05],
        ])

        mask = create_dgpo_reward_mask(scores, response_length=3, reward_threshold=0.1)

        torch.testing.assert_close(
            mask,
            torch.tensor([
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]),
        )

    @staticmethod
    def _batch():
        return DataProto.from_dict(tensors={
            'responses': torch.ones(2, 3, dtype=torch.long),
            'attention_mask': torch.ones(2, 5, dtype=torch.long),
            # The middle response token represents retrieved information.
            'info_mask': torch.tensor([
                [1, 1, 1, 0, 1],
                [1, 1, 1, 0, 1],
            ], dtype=torch.long),
            'old_log_probs': torch.full((2, 3), 0.5),
            'ref_log_prob': torch.zeros(2, 3),
            'token_level_scores': torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        })

    def test_selective_kl_leaves_correct_reward_unregularized(self):
        batch, metrics = apply_kl_penalty(
            self._batch(),
            kl_ctrl=core_algos.FixedKLController(kl_coef=0.001),
            kl_penalty='kl',
            dgpo_selective_kl=True,
            dgpo_reward_threshold=0.1,
        )

        torch.testing.assert_close(
            batch.batch['token_level_rewards'],
            torch.tensor([
                [-0.0005, 0.0, -0.0005],
                [0.0, 0.0, 1.0],
            ]),
        )
        self.assertEqual(metrics['dgpo/incorrect_trajectory_fraction'], 0.5)
        self.assertAlmostEqual(metrics['critic/kl'], 0.25)

    def test_default_kl_path_is_unchanged(self):
        batch, metrics = apply_kl_penalty(
            self._batch(),
            kl_ctrl=core_algos.FixedKLController(kl_coef=0.001),
            kl_penalty='kl',
        )

        torch.testing.assert_close(
            batch.batch['token_level_rewards'],
            torch.tensor([
                [-0.0005, 0.0, -0.0005],
                [-0.0005, 0.0, 0.9995],
            ]),
        )
        self.assertNotIn('dgpo/incorrect_trajectory_fraction', metrics)
        self.assertAlmostEqual(metrics['critic/kl'], 0.5)

    def test_selective_teacher_reward_is_group_normalized_by_grpo(self):
        batch = self._batch()
        batch.non_tensor_batch['uid'] = np.array(['prompt', 'prompt'], dtype=object)
        batch, _ = apply_kl_penalty(
            batch,
            kl_ctrl=core_algos.FixedKLController(kl_coef=0.001),
            kl_penalty='kl',
            dgpo_selective_kl=True,
            dgpo_reward_threshold=0.1,
        )

        output = compute_advantage(batch, adv_estimator='grpo')

        self.assertTrue(torch.all(output.batch['advantages'][0] < 0))
        self.assertTrue(torch.all(output.batch['advantages'][1] > 0))


class DGPOConfigTest(unittest.TestCase):
    @staticmethod
    def _config():
        return OmegaConf.create({
            'do_search': True,
            'algorithm': {
                'adv_estimator': 'gae',
                'kl_penalty': 'kl',
                'kl_ctrl': {'type': 'fixed', 'kl_coef': 0.001},
                'dgpo': {'enable': True, 'reward_threshold': 0.1},
            },
            'actor_rollout_ref': {
                'actor': {
                    'strategy': 'fsdp',
                    'state_masking': True,
                    'use_kl_loss': False,
                },
                'rollout': {'n': 1, 'n_agent': 1},
                'ref': {'model_path': 'teacher'},
            },
        })

    def test_official_dgpo_online_config_is_accepted(self):
        validate_dgpo_config(self._config())

    def test_dgpo_accepts_er_opd_rollout_multiplicity(self):
        config = self._config()
        config.actor_rollout_ref.rollout.n_agent = 8
        validate_dgpo_config(config)

    def test_dgpo_accepts_grpo_with_group_rollouts(self):
        config = self._config()
        config.algorithm.adv_estimator = 'grpo'
        config.actor_rollout_ref.rollout.n_agent = 8
        validate_dgpo_config(config)

    def test_dgpo_rejects_grpo_without_group_rollouts(self):
        config = self._config()
        config.algorithm.adv_estimator = 'grpo'
        with self.assertRaisesRegex(ValueError, 'at least two trajectories'):
            validate_dgpo_config(config)

    def test_dgpo_requires_an_independent_teacher(self):
        config = self._config()
        config.actor_rollout_ref.ref.model_path = None
        with self.assertRaisesRegex(ValueError, 'teacher checkpoint'):
            validate_dgpo_config(config)

    def test_dgpo_rejects_actor_side_uniform_kl(self):
        config = self._config()
        config.actor_rollout_ref.actor.use_kl_loss = True
        with self.assertRaisesRegex(ValueError, 'selective reward-side KL'):
            validate_dgpo_config(config)


if __name__ == '__main__':
    unittest.main()
