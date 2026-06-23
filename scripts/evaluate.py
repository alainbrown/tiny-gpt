import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from tiny_gpt.modeling_tiny_gpt import TinyGPTForCausalLM


DEFAULT_PROMPTS = [
    "Once upon a time",
    "The little fox looked up at the moon and",
    "Mia opened the mysterious blue door.",
    "Tom found a tiny dragon under his bed.",
    "Ella broke her brother's toy and",
    "Write a short bedtime story about a robot who learns to share.",
]

DEFAULT_SETTINGS = [
    {"name": "conservative", "temperature": 0.6, "top_k": 20},
    {"name": "balanced", "temperature": 0.8, "top_k": 40},
    {"name": "creative", "temperature": 1.0, "top_k": 80},
    {"name": "no_top_k", "temperature": 0.8, "top_k": 0},
]

DEFAULT_SEEDS = [1337, 2024, 7]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Tiny GPT storyteller checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tiny_gpt"),
    )
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--eval-stories", type=int, default=5000)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help=(
            "Generation seed. Repeat for multiple seeds. Defaults to "
            f"{DEFAULT_SEEDS}."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Use one custom generation setting with this temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Use one custom generation setting with this top-k value.",
    )
    parser.add_argument(
        "--generation-setting",
        action="append",
        dest="generation_settings",
        metavar="NAME:TEMPERATURE:TOP_K",
        help=(
            "Generation setting to evaluate. Repeat for multiple settings. "
            "Example: balanced:0.8:40. Overrides --temperature/--top-k."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to sample. Repeat the flag to provide multiple prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/storyteller-full-eval.json"),
        help="Path for the JSON evaluation report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Path for the Markdown evaluation report. Defaults to output with .md suffix.",
    )
    return parser.parse_args()


def parse_generation_settings(args):
    if args.generation_settings:
        settings = []
        for raw_setting in args.generation_settings:
            parts = raw_setting.split(":")
            if len(parts) != 3:
                raise ValueError(
                    "--generation-setting must use NAME:TEMPERATURE:TOP_K"
                )
            name, temperature, top_k = parts
            settings.append(
                {
                    "name": name,
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                }
            )
    elif args.temperature is not None or args.top_k is not None:
        settings = [
            {
                "name": "custom",
                "temperature": args.temperature if args.temperature is not None else 0.8,
                "top_k": args.top_k if args.top_k is not None else 40,
            }
        ]
    else:
        settings = DEFAULT_SETTINGS

    for setting in settings:
        if setting["temperature"] <= 0:
            raise ValueError("Generation temperature must be greater than zero")
        if setting["top_k"] < 0:
            raise ValueError("Generation top_k must be non-negative")
    return settings


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_value(args):
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={Path.cwd()}", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_commit_from_files(git_dir=Path(".git")):
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None

    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head

    ref_path = git_dir / head.removeprefix("ref: ")
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()

    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        ref_name = head.removeprefix("ref: ")
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            commit, _, ref = line.partition(" ")
            if ref == ref_name:
                return commit
    return None


def git_metadata():
    status = git_value(["status", "--short"])
    commit = git_value(["rev-parse", "HEAD"]) or git_commit_from_files()
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status": status,
        "dirty_available": status is not None,
    }


def file_sha256(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(checkpoint, device):
    checkpoint = resolve_checkpoint(checkpoint)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(checkpoint / "tokenizer.json"),
        eos_token="<EOS>",
        pad_token="<EOS>",
    )
    model = TinyGPTForCausalLM.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint


def resolve_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    latest = checkpoint / "latest"
    if (latest / "config.json").is_file():
        return latest
    return checkpoint


def load_validation_texts(dataset_name, split, text_column, story_limit):
    dataset = load_dataset(dataset_name, split=split)
    if story_limit is not None:
        dataset = dataset.select(range(min(story_limit, len(dataset))))
    return dataset[text_column], len(dataset)


def make_eval_blocks(texts, tokenizer, context_size):
    encoded = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    tokens = []
    for token_ids in encoded:
        tokens.extend(token_ids)
        tokens.append(tokenizer.eos_token_id)

    block_size = context_size + 1
    usable_tokens = len(tokens) - (len(tokens) % block_size)
    return [
        tokens[offset : offset + block_size]
        for offset in range(0, usable_tokens, block_size)
    ]


def evaluate_loss(model, blocks, batch_size, max_batches, device):
    losses = []
    tokens_evaluated = 0

    with torch.inference_mode():
        for batch_index, offset in enumerate(range(0, len(blocks), batch_size)):
            if batch_index >= max_batches:
                break

            batch = torch.tensor(
                blocks[offset : offset + batch_size],
                dtype=torch.long,
                device=device,
            )
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            logits = model(input_ids=input_ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
            )
            losses.append(loss.item())
            tokens_evaluated += labels.numel()

    if not losses:
        raise ValueError("The evaluation selection produced no complete blocks.")

    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "batches": len(losses),
        "tokens": tokens_evaluated,
    }


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample(model, tokenizer, prompt, max_new_tokens, temperature, top_k, seed, device):
    set_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated_ids = input_ids
    reached_eos = False

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            model_input = generated_ids[:, -model.config.context_size :]
            logits = model(input_ids=model_input).logits[:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                values, _ = torch.topk(logits, k)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_id), dim=1)

            if next_id.item() == tokenizer.eos_token_id:
                reached_eos = True
                break

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text, reached_eos, generated_ids[0].tolist()


def ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def repeated_ngram_rate(tokens, n):
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def distinct_n(tokens, n):
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def sentence_count(text):
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(sentences)


def text_metrics(tokenizer, prompt, text, reached_eos, max_new_tokens):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    continuation_ids = text_ids[len(prompt_ids) :]
    continuation_text = text[len(prompt) :] if text.startswith(prompt) else text
    sentences = sentence_count(continuation_text)

    prompt_token_overlap = 0
    for left, right in zip(prompt_ids, text_ids):
        if left != right:
            break
        prompt_token_overlap += 1

    return {
        "total_tokens": len(text_ids),
        "continuation_tokens": len(continuation_ids),
        "sentence_count": sentences,
        "avg_sentence_tokens": (
            len(continuation_ids) / sentences if sentences > 0 else 0.0
        ),
        "distinct_1": distinct_n(continuation_ids, 1),
        "distinct_2": distinct_n(continuation_ids, 2),
        "repeated_2gram_rate": repeated_ngram_rate(continuation_ids, 2),
        "repeated_3gram_rate": repeated_ngram_rate(continuation_ids, 3),
        "prompt_prefix_token_ratio": (
            prompt_token_overlap / len(prompt_ids) if prompt_ids else 0.0
        ),
        "reached_eos": reached_eos,
        "hit_max_tokens": not reached_eos and len(continuation_ids) >= max_new_tokens,
    }


def summarize_samples(samples):
    by_setting = {}
    for sample_row in samples:
        setting = sample_row["setting"]["name"]
        by_setting.setdefault(setting, []).append(sample_row["metrics"])

    summary = {}
    for setting, metrics in by_setting.items():
        count = len(metrics)
        summary[setting] = {
            "samples": count,
            "avg_continuation_tokens": sum(
                metric["continuation_tokens"] for metric in metrics
            )
            / count,
            "avg_sentence_count": sum(
                metric["sentence_count"] for metric in metrics
            )
            / count,
            "avg_distinct_1": sum(metric["distinct_1"] for metric in metrics)
            / count,
            "avg_distinct_2": sum(metric["distinct_2"] for metric in metrics)
            / count,
            "avg_repeated_2gram_rate": sum(
                metric["repeated_2gram_rate"] for metric in metrics
            )
            / count,
            "avg_repeated_3gram_rate": sum(
                metric["repeated_3gram_rate"] for metric in metrics
            )
            / count,
            "eos_rate": sum(1 for metric in metrics if metric["reached_eos"])
            / count,
            "max_token_rate": sum(
                1 for metric in metrics if metric["hit_max_tokens"]
            )
            / count,
        }
    return summary


def render_markdown(report):
    lines = [
        "# Storyteller Evaluation",
        "",
        "## Metadata",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Dataset: `{report['dataset']['name']}`",
        f"- Split: `{report['dataset']['validation_split']}`",
        f"- Device: `{report['device']}`",
        f"- Timestamp: `{report['timestamp_utc']}`",
        f"- Git commit: `{report['git']['commit']}`",
        f"- Git dirty: `{report['git']['dirty']}`",
        "",
        "## Validation",
        "",
        f"- Loss: `{report['validation']['loss']:.4f}`",
        f"- Perplexity: `{report['validation']['perplexity']:.4f}`",
        f"- Tokens: `{report['validation']['tokens']}`",
        f"- Batches: `{report['validation']['batches']}`",
        "",
        "## Generation Summary",
        "",
        "| Setting | Samples | Avg Tokens | Avg Sentences | Distinct-1 | Distinct-2 | Repeat 2g | Repeat 3g | EOS Rate | Max Token Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for setting, metrics in report["generation"]["summary_by_setting"].items():
        lines.append(
            "| "
            f"{setting} | "
            f"{metrics['samples']} | "
            f"{metrics['avg_continuation_tokens']:.1f} | "
            f"{metrics['avg_sentence_count']:.1f} | "
            f"{metrics['avg_distinct_1']:.3f} | "
            f"{metrics['avg_distinct_2']:.3f} | "
            f"{metrics['avg_repeated_2gram_rate']:.3f} | "
            f"{metrics['avg_repeated_3gram_rate']:.3f} | "
            f"{metrics['eos_rate']:.2f} | "
            f"{metrics['max_token_rate']:.2f} |"
        )

    lines.extend(["", "## Samples", ""])
    for sample_row in report["generation"]["samples"]:
        setting = sample_row["setting"]
        lines.extend(
            [
                (
                    "### "
                    f"{setting['name']} | temp={setting['temperature']} | "
                    f"top_k={setting['top_k']} | seed={sample_row['seed']}"
                ),
                "",
                f"Prompt: `{sample_row['prompt']}`",
                "",
                "Metrics:",
                (
                    f"- tokens={sample_row['metrics']['continuation_tokens']}, "
                    f"sentences={sample_row['metrics']['sentence_count']}, "
                    f"distinct_2={sample_row['metrics']['distinct_2']:.3f}, "
                    f"repeat_3g={sample_row['metrics']['repeated_3gram_rate']:.3f}, "
                    f"eos={sample_row['metrics']['reached_eos']}, "
                    f"hit_max={sample_row['metrics']['hit_max_tokens']}"
                ),
                "",
                "```text",
                sample_row["text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    settings = parse_generation_settings(args)
    seeds = args.seeds or DEFAULT_SEEDS
    prompts = args.prompts or DEFAULT_PROMPTS

    device = select_device()
    model, tokenizer, resolved_checkpoint = load_checkpoint(args.checkpoint, device)
    validation_texts, validation_rows = load_validation_texts(
        args.dataset,
        args.validation_split,
        args.text_column,
        args.eval_stories,
    )
    blocks = make_eval_blocks(
        validation_texts,
        tokenizer,
        model.config.context_size,
    )
    loss_metrics = evaluate_loss(
        model,
        blocks,
        args.batch_size,
        args.eval_batches,
        device,
    )

    samples = []
    for prompt in prompts:
        for setting in settings:
            for seed in seeds:
                text, reached_eos, token_ids = sample(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=setting["temperature"],
                    top_k=setting["top_k"],
                    seed=seed,
                    device=device,
                )
                samples.append(
                    {
                        "prompt": prompt,
                        "setting": setting,
                        "seed": seed,
                        "text": text,
                        "metrics": text_metrics(
                            tokenizer,
                            prompt,
                            text,
                            reached_eos,
                            args.max_new_tokens,
                        ),
                        "generated_token_count": len(token_ids),
                    }
                )

    config_path = resolved_checkpoint / "config.json"
    tokenizer_path = resolved_checkpoint / "tokenizer.json"
    report = {
        "checkpoint": str(args.checkpoint),
        "resolved_checkpoint": str(resolved_checkpoint),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "git": git_metadata(),
        "model_config": read_json(config_path),
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": file_sha256(tokenizer_path),
        },
        "dataset": {
            "name": args.dataset,
            "validation_split": args.validation_split,
            "text_column": args.text_column,
            "requested_eval_stories": args.eval_stories,
            "loaded_eval_stories": validation_rows,
        },
        "validation": loss_metrics,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "prompts": prompts,
            "settings": settings,
            "seeds": seeds,
            "sample_count": len(samples),
            "summary_by_setting": summarize_samples(samples),
            "samples": samples,
        },
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")

    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
