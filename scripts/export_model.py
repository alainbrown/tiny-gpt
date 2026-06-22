import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from tiny_gpt.configuration_tiny_gpt import TinyGPTConfig
from tiny_gpt.modeling_tiny_gpt import TinyGPTForCausalLM


MODEL_CARD = """---
language:
  - en
license: mit
library_name: transformers
pipeline_tag: text-generation
datasets:
  - roneneldan/TinyStories
tags:
  - custom_code
  - educational
---

# Tiny GPT

Tiny GPT is an educational decoder-only Transformer trained from scratch on
the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset. The implementation is intentionally small and readable.

## Model details

- Architecture: decoder-only causal language model
- Context length: 512 tokens
- Vocabulary size: 10,000
- Hidden size: 256
- Transformer layers: 6
- Attention heads: 8

Source code: https://github.com/alainbrown/tiny-gpt

## Usage

This repository contains custom Transformers code. Review it before enabling
`trust_remote_code`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "alainbrown/tiny-gpt"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

inputs = tokenizer("Once upon a time", return_tensors="pt")
logits = model(**inputs).logits
```

## Intended use

This model is intended for education and experimentation. It is not intended
for production, factual question answering, or safety-critical applications.

## Limitations

The model is small, trained on synthetic children's stories, and has not been
comprehensively evaluated. It may produce incoherent, repetitive, incorrect,
or inappropriate text. English is the only supported language.

## Training

The training pipeline is available in the linked GitHub repository. This model
repository excludes optimizer and progress state and contains inference files
only.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Tiny GPT checkpoint for the Hugging Face Hub."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/tiny_gpt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/tiny_gpt_hub")
    )
    return parser.parse_args()


def export_model(checkpoint, output):
    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {', '.join(missing)}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    TinyGPTConfig.register_for_auto_class()
    TinyGPTForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = TinyGPTForCausalLM.from_pretrained(checkpoint)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(checkpoint / "tokenizer.json"),
        eos_token="<EOS>",
        pad_token="<EOS>",
        model_max_length=model.config.context_size,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if "auto_map" not in config:
        raise RuntimeError("Exported config does not contain AutoClass metadata")

    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        output, trust_remote_code=True
    )
    reloaded_model = AutoModelForCausalLM.from_pretrained(
        output, trust_remote_code=True
    )
    inputs = reloaded_tokenizer("Once upon a time", return_tensors="pt")
    outputs = reloaded_model(**inputs)
    expected_shape = (*inputs["input_ids"].shape, reloaded_model.config.vocab_size)
    if tuple(outputs.logits.shape) != expected_shape:
        raise RuntimeError(f"Unexpected logits shape: {tuple(outputs.logits.shape)}")

    generated = reloaded_model.generate(**inputs, max_new_tokens=1)
    if generated.shape[1] != inputs["input_ids"].shape[1] + 1:
        raise RuntimeError("Generation validation returned an unexpected shape")

    excluded = {"optimizer.pt", "TinyStories.progress"}
    leaked = excluded.intersection(path.name for path in output.iterdir())
    if leaked:
        raise RuntimeError(f"Training-only files leaked into export: {sorted(leaked)}")

    print(f"Exported and validated: {output}")


if __name__ == "__main__":
    args = parse_args()
    export_model(args.checkpoint, args.output)
