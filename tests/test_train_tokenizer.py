import tempfile
import unittest
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from scripts.train_tokenizer import train_tokenizer


class TrainTokenizerTest(unittest.TestCase):
    def test_trains_hugging_face_compatible_byte_level_bpe(self):
        dataset = Dataset.from_dict(
            {
                "text": [
                    "Once upon a time, there was a small fox.",
                    "The fox found a bright red kite.",
                    "They played together under the blue sky.",
                ]
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tokenizer.json"
            train_tokenizer(
                dataset=dataset,
                text_column="text",
                vocab_size=300,
                min_frequency=1,
                eos_token="<EOS>",
                output=output,
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(output),
                eos_token="<EOS>",
            )

            encoded = tokenizer(
                "The fox found a kite.",
                add_special_tokens=False,
            )["input_ids"]
            decoded = tokenizer.decode(encoded)

        self.assertTrue(encoded)
        self.assertEqual(decoded, "The fox found a kite.")
        self.assertIsNotNone(tokenizer.eos_token_id)

    def test_rejects_missing_text_column(self):
        dataset = Dataset.from_dict({"content": ["story"]})

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Available columns"):
                train_tokenizer(
                    dataset=dataset,
                    text_column="text",
                    vocab_size=300,
                    min_frequency=1,
                    eos_token="<EOS>",
                    output=Path(directory) / "tokenizer.json",
                )


if __name__ == "__main__":
    unittest.main()
