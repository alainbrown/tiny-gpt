import unittest

import torch

from tiny_gpt.configuration_tiny_gpt import TinyGPTConfig
from tiny_gpt.model import GPTModel, Model, MultiHeadAttention
from tiny_gpt.modeling_tiny_gpt import TinyGPTForCausalLM


class GPTModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPTModel(
            context_size=8,
            vocab_size=32,
            d_model=16,
            n_layers=2,
            n_heads=4,
            dropout=0.0,
        )
        self.model.eval()

    def test_model_alias_is_optimized_model(self):
        self.assertIs(Model, GPTModel)

    def test_output_shape(self):
        input_ids = torch.randint(0, 32, (3, 6))

        logits = self.model(input_ids)

        self.assertEqual(logits.shape, (3, 6, 32))

    def test_rejects_input_beyond_context_size(self):
        input_ids = torch.randint(0, 32, (1, 9))

        with self.assertRaisesRegex(AssertionError, "context_size"):
            self.model(input_ids)

    def test_future_tokens_do_not_change_prefix_logits(self):
        first = torch.tensor([[1, 2, 3, 4, 5, 6]])
        second = torch.tensor([[1, 2, 3, 20, 21, 22]])

        with torch.inference_mode():
            first_logits = self.model(first)
            second_logits = self.model(second)

        torch.testing.assert_close(
            first_logits[:, :3],
            second_logits[:, :3],
            rtol=0,
            atol=0,
        )


class MultiHeadAttentionTest(unittest.TestCase):
    def test_split_and_combine_heads_are_inverse_operations(self):
        attention = MultiHeadAttention(
            d_model=16,
            n_heads=4,
            dropout=0.0,
        )
        inputs = torch.randn(2, 5, 16)

        split = attention.split_heads(inputs)
        combined = attention.combine_heads(split)

        self.assertEqual(split.shape, (2, 4, 5, 4))
        torch.testing.assert_close(combined, inputs)


class TinyGPTForCausalLMTest(unittest.TestCase):
    def test_input_and_output_embeddings_are_tied_by_default(self):
        model = TinyGPTForCausalLM(
            TinyGPTConfig(
                context_size=8,
                vocab_size=32,
                d_model=16,
                n_layers=2,
                n_heads=4,
                dropout=0.0,
            )
        )

        self.assertTrue(model.config.tie_word_embeddings)
        self.assertEqual(
            model.core_model.token_embedding.weight.data_ptr(),
            model.core_model.linear.weight.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
