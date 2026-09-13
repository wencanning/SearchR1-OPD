"""Check that SFT supervision follows target tokens at masked boundaries."""
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer


class SFTLossAlignmentTests(unittest.TestCase):
    def test_only_agent_targets_and_eos_receive_gradients(self):
        # prompt, agent, information, agent, eos, padding
        ids = torch.tensor([[0, 1, 2, 3, 4, 0]])
        mask = torch.tensor([[0, 1, 0, 1, 1, 0]])
        logits = torch.arange(30, dtype=torch.float32).reshape(1, 6, 5).requires_grad_()
        trainer = SimpleNamespace(
            fsdp_model=lambda **kwargs: SimpleNamespace(logits=logits),
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=5)),
            config=SimpleNamespace(data=SimpleNamespace(balance_dp_token=False)),
        )
        batch = dict(input_ids=ids, loss_mask=mask,
                     attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
                     position_ids=torch.arange(6).unsqueeze(0))
        with patch.object(torch.Tensor, 'cuda', lambda t: t), \
                patch('torch.autocast', lambda **kwargs: contextlib.nullcontext()):
            loss = FSDPSFTTrainer._compute_loss(trainer, batch)
        expected = torch.nn.functional.cross_entropy(logits[0, [0, 2, 3]], ids[0, [1, 3, 4]])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum(-1).gt(0).tolist(),
                         [[True, False, True, True, False, False]])


if __name__ == '__main__':
    unittest.main()
