import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tiny_gpt.trainer import Trainer, TrainerConfig


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=7)
        self.embedding = nn.Embedding(7, 4)
        self.output = nn.Linear(4, 7)
        self.tie_weights_calls = 0

    def tie_weights(self):
        self.tie_weights_calls += 1

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def make_trainer(model, grad_accum_steps=1, max_steps=10):
    config = TrainerConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        grad_clip=0.0,
        grad_accum_steps=grad_accum_steps,
        warmup_steps=1,
        max_steps=max_steps,
        min_lr=1e-3,
        eval_interval=10,
    )
    with patch("torch.cuda.is_available", return_value=False):
        return Trainer(model, config)


class TrainerTest(unittest.TestCase):
    def test_initialization_does_not_retie_model_weights(self):
        model = TinyCausalLM()

        make_trainer(model)

        self.assertEqual(model.tie_weights_calls, 0)

    def test_partial_accumulation_matches_full_accumulation(self):
        torch.manual_seed(7)
        full_model = TinyCausalLM()
        partial_model = deepcopy(full_model)
        batches = [
            (
                torch.tensor([[0, 1, 2]]),
                torch.tensor([[1, 2, 3]]),
            ),
            (
                torch.tensor([[3, 4, 5]]),
                torch.tensor([[4, 5, 6]]),
            ),
        ]

        full_trainer = make_trainer(full_model, grad_accum_steps=2)
        partial_trainer = make_trainer(partial_model, grad_accum_steps=4)

        _, _, full_step = full_trainer.train_steps(
            iter(batches),
            num_steps=2,
        )
        _, _, partial_step = partial_trainer.train_steps(
            iter(batches),
            num_steps=4,
        )

        self.assertEqual(full_step, 1)
        self.assertEqual(partial_step, 1)
        for full_parameter, partial_parameter in zip(
            full_model.parameters(),
            partial_model.parameters(),
        ):
            torch.testing.assert_close(
                full_parameter,
                partial_parameter,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_save_and_load_state_restores_step_and_rng(self):
        torch.manual_seed(11)
        trainer = make_trainer(TinyCausalLM())

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            trainer.save_state(checkpoint_dir, global_step=37)
            expected_random = torch.rand(4)

            torch.manual_seed(99)
            restored_trainer = make_trainer(TinyCausalLM())
            restored_step = restored_trainer.load_state(checkpoint_dir)
            actual_random = torch.rand(4)

        self.assertEqual(restored_step, 37)
        torch.testing.assert_close(actual_random, expected_random)

    def test_max_steps_is_a_hard_optimizer_step_limit(self):
        batches = [
            (
                torch.tensor([[0, 1, 2]]),
                torch.tensor([[1, 2, 3]]),
            )
            for _ in range(10)
        ]
        trainer = make_trainer(
            TinyCausalLM(),
            grad_accum_steps=2,
            max_steps=2,
        )

        losses, _, global_step = trainer.train_steps(
            iter(batches),
            num_steps=10,
        )

        self.assertEqual(global_step, 2)
        self.assertEqual(len(losses), 4)

    def test_train_steps_returns_observability_stats(self):
        batches = [
            (
                torch.tensor([[0, 1, 2]]),
                torch.tensor([[1, 2, 3]]),
            )
            for _ in range(3)
        ]
        trainer = make_trainer(
            TinyCausalLM(),
            grad_accum_steps=1,
            max_steps=3,
        )

        stats = trainer.train_steps(
            iter(batches),
            num_steps=3,
        )

        self.assertEqual(stats.global_step, 3)
        self.assertEqual(stats.optimizer_steps, 3)
        self.assertGreater(stats.tokens_processed, 0)
        self.assertEqual(len(stats.grad_norm_history), 3)
        self.assertTrue(all(value >= 0 for value in stats.grad_norm_history))


if __name__ == "__main__":
    unittest.main()
