import argparse
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Rust-backed byte-level BPE tokenizer."
    )
    parser.add_argument(
        "--dataset",
        default="roneneldan/TinyStories",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-config",
        help="Optional Hugging Face dataset configuration name.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--num-examples",
        type=int,
        help="Optional example limit. The complete split is used by default.",
    )
    parser.add_argument("--eos-token", default="<EOS>")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/tiny_gpt/tokenizer.json"),
    )
    return parser.parse_args()


def iter_texts(dataset, text_column, num_examples=None):
    rows = dataset if num_examples is None else islice(dataset, num_examples)
    for row in rows:
        text = row[text_column]
        if not isinstance(text, str):
            raise TypeError(
                f"Column {text_column!r} must contain strings, "
                f"got {type(text).__name__}"
            )
        yield text


def train_tokenizer(
    dataset,
    text_column,
    vocab_size,
    min_frequency,
    eos_token,
    output,
    num_examples=None,
):
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Column {text_column!r} not found. "
            f"Available columns: {dataset.column_names}"
        )
    if vocab_size <= len(pre_tokenizers.ByteLevel.alphabet()) + 1:
        raise ValueError("vocab_size must exceed the byte alphabet plus EOS")
    if num_examples is not None and num_examples <= 0:
        raise ValueError("num_examples must be positive")

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[eos_token],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    length = len(dataset)
    if num_examples is not None:
        length = min(length, num_examples)

    tokenizer.train_from_iterator(
        iter_texts(dataset, text_column, num_examples),
        trainer=trainer,
        length=length,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer


def main(args):
    print(f"Loading {args.dataset} ({args.split})...")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
    )

    example_count = len(dataset)
    if args.num_examples is not None:
        example_count = min(example_count, args.num_examples)
    print(
        f"Training byte-level BPE on {example_count:,} examples "
        f"from column {args.text_column!r}..."
    )

    tokenizer = train_tokenizer(
        dataset=dataset,
        text_column=args.text_column,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        eos_token=args.eos_token,
        output=args.output,
        num_examples=args.num_examples,
    )
    print(
        f"Saved {tokenizer.get_vocab_size():,}-token tokenizer "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main(parse_args())
