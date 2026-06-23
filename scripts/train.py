import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from tiny_gpt.dataset import PackedDataset
from tiny_gpt.hf_model import TinyGPTForCausalLM, TinyGPTConfig
from tiny_gpt.trainer import Trainer, TrainerConfig


DEFAULT_SAMPLE_PROMPTS = [
    "Once upon a time",
    "The little fox looked up at the moon and",
    "Mia opened the mysterious blue door.",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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

    ref_name = head.removeprefix("ref: ")
    ref_path = git_dir / ref_name
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()

    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
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
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_aim_run(args, metadata):
    if args.disable_aim:
        return None

    try:
        from aim import Run
    except ImportError:
        print("Aim is not installed; continuing without Aim tracking.")
        return None

    run = Run(repo=args.aim_repo, experiment=args.aim_experiment)
    run["hparams"] = vars(args)
    run["metadata"] = metadata
    return run


def track_metrics(aim_run, metrics, step, context):
    if aim_run is None:
        return
    for name, value in metrics.items():
        if value is None:
            continue
        aim_run.track(value, name=name, step=step, context=context)


def track_text(aim_run, text, name, step, context):
    if aim_run is None:
        return
    try:
        from aim import Text
    except ImportError:
        return
    aim_run.track(Text(text), name=name, step=step, context=context)


def checkpoint_path(checkpoint_dir, kind):
    root = Path(checkpoint_dir)
    if kind == "legacy":
        return root
    return root / kind


def resolve_resume_checkpoint(checkpoint_dir):
    root = Path(checkpoint_dir)
    latest = checkpoint_path(root, "latest")
    if (latest / "config.json").is_file():
        return latest
    if (root / "config.json").is_file():
        return root
    return None


def progress_candidates(checkpoint_dir, resume_checkpoint):
    paths = []
    if resume_checkpoint is not None:
        paths.append(Path(resume_checkpoint) / "training_progress.json")
    paths.append(Path(checkpoint_dir) / "training_progress.json")
    return paths


def save_progress(path, epoch, epochs, start_story, global_step, dataset_size):
    write_json(
        path,
        {
            "epoch": epoch,
            "epochs": epochs,
            "start_story": start_story,
            "global_step": global_step,
            "stories_seen": epoch * dataset_size + min(start_story, dataset_size),
        },
    )


def save_checkpoint(
    model,
    tokenizer,
    trainer,
    checkpoint_dir,
    kind,
    global_step,
    metadata,
    progress,
    val_loss=None,
):
    path = checkpoint_path(checkpoint_dir, kind)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    trainer.save_state(path, global_step)
    write_json(path / "training_progress.json", progress)
    checkpoint_metadata = {
        **metadata,
        "checkpoint": {
            "kind": kind,
            "path": str(path),
            "global_step": global_step,
            "val_loss": val_loss,
            "saved_at": utc_now(),
        },
        "progress": progress,
    }
    write_json(path / "run_metadata.json", checkpoint_metadata)
    return path


def save_step_checkpoint(
    model,
    tokenizer,
    trainer,
    checkpoint_dir,
    global_step,
    metadata,
    progress,
    val_loss,
    keep_last,
):
    if keep_last <= 0:
        return None

    step_root = Path(checkpoint_dir) / "steps"
    step_root.mkdir(parents=True, exist_ok=True)
    path = step_root / f"step_{global_step:08d}"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    trainer.save_state(path, global_step)
    write_json(path / "training_progress.json", progress)
    write_json(
        path / "run_metadata.json",
        {
            **metadata,
            "checkpoint": {
                "kind": "step",
                "path": str(path),
                "global_step": global_step,
                "val_loss": val_loss,
                "saved_at": utc_now(),
            },
            "progress": progress,
        },
    )

    step_dirs = sorted(
        [item for item in step_root.iterdir() if item.is_dir()],
        key=lambda item: item.stat().st_mtime,
    )
    while len(step_dirs) > keep_last:
        shutil.rmtree(step_dirs.pop(0))
    return path


def pack_texts(texts, tokenizer, context_size):
    if not isinstance(texts, list):
        texts = list(texts)
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    concatenated = []
    for ids in encoded:
        concatenated.extend(ids)
        concatenated.append(tokenizer.eos_token_id)

    block_size = context_size + 1
    total_length = (len(concatenated) // block_size) * block_size
    return [
        concatenated[i : i + block_size]
        for i in range(0, total_length, block_size)
    ]


def create_train_loader(
    dataset,
    tokenizer,
    text_column,
    context_size,
    batch_size,
    n_examples,
    start_example=0,
):
    end_example = min(start_example + n_examples, len(dataset))
    texts = dataset[start_example:end_example][text_column]
    print(f"Tokenizing chunk of {len(texts)} stories in bulk...")
    print("Packing tokens into blocks...")
    blocks = pack_texts(texts, tokenizer, context_size)
    train_ds = PackedDataset(blocks)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )


def create_validation_loader(
    dataset,
    tokenizer,
    text_column,
    context_size,
    batch_size,
):
    print(f"Tokenizing {len(dataset)} validation stories...")
    blocks = pack_texts(
        dataset[text_column],
        tokenizer,
        context_size,
    )
    return DataLoader(
        PackedDataset(blocks),
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
    )


def sample_text(model, tokenizer, prompt, args, device, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated_ids = input_ids

    model.eval()
    with torch.inference_mode():
        for _ in range(args.sample_max_new_tokens):
            model_input = generated_ids[:, -model.config.context_size :]
            logits = model(input_ids=model_input).logits[:, -1, :]
            logits = logits / args.sample_temperature

            if args.sample_top_k > 0:
                top_k = min(args.sample_top_k, logits.shape[-1])
                values, _ = torch.topk(logits, top_k)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_id), dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    model.train()
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def write_training_samples(samples_path, samples):
    path = Path(samples_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample) + "\n")


def model_arg_config(args, vocab_size):
    return {
        "context_size": args.context_size,
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
    }


def dataset_metadata(args, train_dataset, validation_dataset):
    return {
        "name": args.dataset,
        "config": args.dataset_config,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "text_column": args.text_column,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
    }


def environment_metadata():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def build_metadata(args, train_dataset, validation_dataset, tokenizer, train_config):
    return {
        "created_at": utc_now(),
        "args": vars(args),
        "git": git_metadata(),
        "dataset": dataset_metadata(args, train_dataset, validation_dataset),
        "tokenizer": {
            "path": args.tokenizer_path,
            "sha256": file_sha256(args.tokenizer_path),
            "vocab_size": len(tokenizer),
        },
        "requested_model_config": model_arg_config(args, len(tokenizer)),
        "trainer_config": train_config.__dict__,
        "environment": environment_metadata(),
    }


def validate_resume_metadata(args, tokenizer, metadata, allow_mismatch):
    if metadata is None:
        print("No prior run metadata found; resume compatibility checks skipped.")
        return

    mismatches = []
    expected = {
        "dataset.name": args.dataset,
        "dataset.config": args.dataset_config,
        "dataset.train_split": args.train_split,
        "dataset.validation_split": args.validation_split,
        "dataset.text_column": args.text_column,
        "tokenizer.sha256": file_sha256(args.tokenizer_path),
        "requested_model_config.context_size": args.context_size,
        "requested_model_config.vocab_size": len(tokenizer),
    }

    for key, expected_value in expected.items():
        cursor = metadata
        for part in key.split("."):
            cursor = cursor.get(part) if isinstance(cursor, dict) else None
        if cursor != expected_value:
            mismatches.append((key, cursor, expected_value))

    if mismatches and not allow_mismatch:
        rendered = "\n".join(
            f"- {key}: checkpoint={actual!r}, current={expected!r}"
            for key, actual, expected in mismatches
        )
        raise ValueError(
            "Checkpoint metadata does not match current training arguments. "
            "Use --allow_config_mismatch only if this is intentional.\n"
            + rendered
        )
    if mismatches:
        print("WARNING: resume metadata mismatches allowed:")
        for key, actual, expected in mismatches:
            print(f"  - {key}: checkpoint={actual!r}, current={expected!r}")


def progress_payload(epoch, epochs, start_story, global_step, dataset_size):
    return {
        "epoch": epoch,
        "epochs": epochs,
        "start_story": start_story,
        "global_step": global_step,
        "stories_seen": epoch * dataset_size + min(start_story, dataset_size),
    }


def load_progress(checkpoint_dir, resume_checkpoint, train_size):
    epoch = 0
    start_story = 0
    global_step = 0
    progress_file = None

    for candidate in progress_candidates(checkpoint_dir, resume_checkpoint):
        if candidate.is_file():
            progress_file = candidate
            content = candidate.read_text(encoding="utf-8").strip()
            if "{" in content:
                state = json.loads(content)
                epoch = state.get("epoch", 0)
                start_story = state.get("start_story", 0)
                global_step = state.get("global_step", 0)
            else:
                start_story = int(content)
            break

    if epoch == 0 and start_story >= train_size:
        epoch = 1
        start_story = 0

    return epoch, start_story, global_step, progress_file


def log_checkpoint_event(aim_run, kind, path, global_step, val_loss, context):
    track_metrics(
        aim_run,
        {
            "checkpoint/save_event": 1,
            f"checkpoint/{kind}_save_event": 1,
            "checkpoint/global_step": global_step,
            "checkpoint/val_loss": val_loss,
        },
        global_step,
        context,
    )
    if aim_run is not None:
        aim_run[f"checkpoint/{kind}_path"] = str(path)


def main(args):
    if not os.path.isfile(args.tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {args.tokenizer_path}. "
            "Run scripts/train_tokenizer.py first."
        )
    if args.sample_temperature <= 0:
        raise ValueError("--sample_temperature must be greater than zero")
    if args.sample_top_k < 0:
        raise ValueError("--sample_top_k must be non-negative")

    print(f"Loading {args.dataset}...")
    train_dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.train_split,
    )
    validation_dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.validation_split,
    )
    for split_name, dataset in (
        (args.train_split, train_dataset),
        (args.validation_split, validation_dataset),
    ):
        if args.text_column not in dataset.column_names:
            raise ValueError(
                f"Column {args.text_column!r} is missing from split "
                f"{split_name!r}. Available columns: {dataset.column_names}"
            )

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_path,
        eos_token="<EOS>",
    )

    resume_checkpoint = resolve_resume_checkpoint(args.checkpoint_dir)
    epoch, start_story, global_step, progress_file = load_progress(
        args.checkpoint_dir,
        resume_checkpoint,
        len(train_dataset),
    )
    if progress_file is not None:
        print(
            f"Resuming from epoch {epoch + 1}/{args.epochs}, "
            f"story {start_story}, global step {global_step}..."
        )

    if resume_checkpoint is not None:
        print(f"Resuming model from {resume_checkpoint}...")
        model = TinyGPTForCausalLM.from_pretrained(resume_checkpoint)
    else:
        print("Initializing new Model...")
        config = TinyGPTConfig(**model_arg_config(args, len(tokenizer)))
        model = TinyGPTForCausalLM(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}")

    train_config = TrainerConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_decay_steps=args.lr_decay_steps,
        min_lr=args.min_lr,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_dir=args.checkpoint_dir,
    )

    prior_metadata = (
        read_json(Path(resume_checkpoint) / "run_metadata.json")
        if resume_checkpoint is not None
        else None
    )
    validate_resume_metadata(
        args,
        tokenizer,
        prior_metadata,
        args.allow_config_mismatch,
    )

    metadata = build_metadata(
        args,
        train_dataset,
        validation_dataset,
        tokenizer,
        train_config,
    )
    write_json(Path(args.checkpoint_dir) / "run_metadata.json", metadata)

    trainer = Trainer(model=model, config=train_config)
    if resume_checkpoint is not None:
        restored_step = trainer.load_state(resume_checkpoint)
        if restored_step is not None:
            print("Resuming trainer state...")
            global_step = restored_step

    aim_run = create_aim_run(args, metadata)

    val_loader = create_validation_loader(
        dataset=validation_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
        context_size=args.context_size,
        batch_size=args.batch_size,
    )

    start_time = time.perf_counter()
    total_stories = len(train_dataset) * args.epochs
    best_val_loss = (
        prior_metadata.get("summary", {}).get("best_val_loss")
        if isinstance(prior_metadata, dict)
        else None
    )
    best_global_step = (
        prior_metadata.get("summary", {}).get("best_global_step")
        if isinstance(prior_metadata, dict)
        else None
    )
    total_tokens_processed = 0
    warnings = []

    print(
        f"Starting training for {args.epochs} epoch(s), "
        f"in chunks of {args.chunk_size} stories..."
    )
    if args.max_steps is not None:
        print(f"Optional safety cap active: max_steps={args.max_steps}")

    try:
        while epoch < args.epochs:
            if trainer.reached_max_steps(global_step):
                print(
                    f"Reached optional max_steps={args.max_steps}. "
                    f"Checkpoint saved at global step {global_step}."
                )
                break

            if start_story >= len(train_dataset):
                epoch += 1
                start_story = 0
                progress = progress_payload(
                    epoch,
                    args.epochs,
                    start_story,
                    global_step,
                    len(train_dataset),
                )
                latest_path = save_checkpoint(
                    model,
                    tokenizer,
                    trainer,
                    args.checkpoint_dir,
                    "latest",
                    global_step,
                    metadata,
                    progress,
                    best_val_loss,
                )
                write_json(Path(args.checkpoint_dir) / "training_progress.json", progress)
                log_checkpoint_event(
                    aim_run,
                    "latest",
                    latest_path,
                    global_step,
                    best_val_loss,
                    {"epoch": epoch},
                )
                print(f"Finished epoch {epoch}/{args.epochs}. Global step: {global_step}")
                continue

            end_story = min(start_story + args.chunk_size, len(train_dataset))
            print(
                f"\n--- Epoch {epoch + 1}/{args.epochs} | "
                f"stories {start_story} to {end_story} ---"
            )
            train_loader = create_train_loader(
                dataset=train_dataset,
                tokenizer=tokenizer,
                text_column=args.text_column,
                context_size=args.context_size,
                batch_size=args.batch_size,
                n_examples=args.chunk_size,
                start_example=start_story,
            )

            train_iter = iter(train_loader)

            while True:
                stats = trainer.train_steps(
                    train_iter,
                    num_steps=args.eval_interval,
                    global_step=global_step,
                )
                train_losses = stats.losses
                global_step = stats.global_step
                total_tokens_processed += stats.tokens_processed

                if len(train_losses) == 0:
                    print("Chunk exhausted!")
                    break

                train_loss = statistics.mean(train_losses)
                train_perplexity = math.exp(train_loss)
                tokens_per_second = (
                    statistics.mean(stats.tokens_per_sec_history)
                    if stats.tokens_per_sec_history
                    else 0.0
                )
                mean_grad_norm = (
                    statistics.mean(stats.grad_norm_history)
                    if stats.grad_norm_history
                    else 0.0
                )
                max_grad_norm = (
                    max(stats.grad_norm_history)
                    if stats.grad_norm_history
                    else 0.0
                )
                last_grad_norm = (
                    stats.grad_norm_history[-1]
                    if stats.grad_norm_history
                    else 0.0
                )
                clipping_rate = (
                    stats.clipped_steps / stats.optimizer_steps
                    if stats.optimizer_steps
                    else 0.0
                )
                val_iter = iter(val_loader)
                val_loss = trainer.estimate_loss(val_iter)
                val_perplexity = math.exp(val_loss)
                val_gap = val_loss - train_loss
                elapsed_seconds = time.perf_counter() - start_time
                stories_seen = epoch * len(train_dataset) + end_story
                story_fraction = stories_seen / total_stories if total_stories else 0.0
                eta_seconds = (
                    elapsed_seconds * (1.0 - story_fraction) / story_fraction
                    if story_fraction > 0
                    else None
                )
                epoch_fraction = end_story / len(train_dataset)

                is_best = best_val_loss is None or val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_global_step = global_step

                eta_text = (
                    f"{eta_seconds / 60:.1f}m"
                    if eta_seconds is not None
                    else "unknown"
                )
                print(
                    f"*** EVALUATION | epoch {epoch + 1}/{args.epochs} | "
                    f"global step {global_step} | "
                    f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
                    f"gap: {val_gap:.4f} | grad norm: {mean_grad_norm:.3f} | "
                    f"{tokens_per_second:.0f} tok/s | eta: {eta_text} ***"
                )

                progress = progress_payload(
                    epoch,
                    args.epochs,
                    start_story,
                    global_step,
                    len(train_dataset),
                )
                context = {"epoch": epoch + 1}
                metrics = {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/perplexity": train_perplexity,
                    "val/perplexity": val_perplexity,
                    "train/val_loss_gap": val_gap,
                    "train/grad_norm_mean": mean_grad_norm,
                    "train/grad_norm_max": max_grad_norm,
                    "train/grad_norm_last": last_grad_norm,
                    "train/grad_clip_rate": clipping_rate,
                    "train/learning_rate": trainer.get_lr(global_step),
                    "perf/tokens_per_second": tokens_per_second,
                    "progress/tokens_processed": total_tokens_processed,
                    "progress/stories_seen": stories_seen,
                    "progress/story_index": end_story,
                    "progress/epoch": epoch + 1,
                    "progress/epoch_fraction": epoch_fraction,
                    "time/elapsed_seconds": elapsed_seconds,
                    "time/eta_seconds": eta_seconds,
                    "checkpoint/is_best": 1 if is_best else 0,
                    "checkpoint/best_val_loss": best_val_loss,
                    "checkpoint/best_global_step": best_global_step,
                }
                track_metrics(aim_run, metrics, global_step, context)

                latest_path = save_checkpoint(
                    model,
                    tokenizer,
                    trainer,
                    args.checkpoint_dir,
                    "latest",
                    global_step,
                    {
                        **metadata,
                        "summary": {
                            "best_val_loss": best_val_loss,
                            "best_global_step": best_global_step,
                        },
                    },
                    progress,
                    val_loss,
                )
                log_checkpoint_event(
                    aim_run,
                    "latest",
                    latest_path,
                    global_step,
                    val_loss,
                    context,
                )

                if is_best:
                    best_path = save_checkpoint(
                        model,
                        tokenizer,
                        trainer,
                        args.checkpoint_dir,
                        "best",
                        global_step,
                        {
                            **metadata,
                            "summary": {
                                "best_val_loss": best_val_loss,
                                "best_global_step": best_global_step,
                            },
                        },
                        progress,
                        val_loss,
                    )
                    log_checkpoint_event(
                        aim_run,
                        "best",
                        best_path,
                        global_step,
                        val_loss,
                        context,
                    )

                step_path = save_step_checkpoint(
                    model,
                    tokenizer,
                    trainer,
                    args.checkpoint_dir,
                    global_step,
                    metadata,
                    progress,
                    val_loss,
                    args.keep_step_checkpoints,
                )
                if step_path is not None:
                    log_checkpoint_event(
                        aim_run,
                        "step",
                        step_path,
                        global_step,
                        val_loss,
                        context,
                    )

                write_json(Path(args.checkpoint_dir) / "training_progress.json", progress)

                if not args.disable_sample_logging:
                    sample_prompts = args.sample_prompt or DEFAULT_SAMPLE_PROMPTS
                    samples = []
                    for sample_index, prompt in enumerate(sample_prompts):
                        sample = {
                            "timestamp_utc": utc_now(),
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "prompt": prompt,
                            "text": sample_text(
                                model,
                                tokenizer,
                                prompt,
                                args,
                                device,
                                args.sample_seed + sample_index,
                            ),
                            "temperature": args.sample_temperature,
                            "top_k": args.sample_top_k,
                        }
                        samples.append(sample)
                        track_text(
                            aim_run,
                            sample["text"],
                            "generation/sample",
                            global_step,
                            {**context, "prompt": prompt},
                        )
                    write_training_samples(args.samples_path, samples)
                    track_metrics(
                        aim_run,
                        {"generation/sample_count": len(samples)},
                        global_step,
                        context,
                    )

                print(f"Recovery checkpoint saved at global step {global_step}.")

                if trainer.reached_max_steps(global_step):
                    print(
                        f"Reached optional max_steps={args.max_steps}. "
                        f"Checkpoint saved at global step {global_step}."
                    )
                    break

                if len(train_losses) < args.eval_interval:
                    print("Chunk exhausted!")
                    break

            start_story = end_story
            progress = progress_payload(
                epoch,
                args.epochs,
                start_story,
                global_step,
                len(train_dataset),
            )
            latest_path = save_checkpoint(
                model,
                tokenizer,
                trainer,
                args.checkpoint_dir,
                "latest",
                global_step,
                {
                    **metadata,
                    "summary": {
                        "best_val_loss": best_val_loss,
                        "best_global_step": best_global_step,
                    },
                },
                progress,
                best_val_loss,
            )
            write_json(Path(args.checkpoint_dir) / "training_progress.json", progress)
            log_checkpoint_event(
                aim_run,
                "latest",
                latest_path,
                global_step,
                best_val_loss,
                {"epoch": epoch + 1},
            )

            print(
                f"Checkpoint saved. Epoch {epoch + 1}/{args.epochs}, "
                f"progress: {start_story}/{len(train_dataset)} stories. "
                f"Global step: {global_step}"
            )

            if trainer.reached_max_steps(global_step):
                break

    except FloatingPointError as error:
        warnings.append(str(error))
        emergency_progress = progress_payload(
            epoch,
            args.epochs,
            start_story,
            global_step,
            len(train_dataset),
        )
        emergency_path = save_checkpoint(
            model,
            tokenizer,
            trainer,
            args.checkpoint_dir,
            "emergency",
            global_step,
            metadata,
            emergency_progress,
            best_val_loss,
        )
        print(f"Emergency checkpoint saved at {emergency_path}")
        raise

    elapsed_seconds = time.perf_counter() - start_time
    final_summary = {
        "finished_at": utc_now(),
        "global_step": global_step,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "best_global_step": best_global_step,
        "tokens_processed": total_tokens_processed,
        "stories_processed": min(total_stories, args.epochs * len(train_dataset)),
        "elapsed_seconds": elapsed_seconds,
        "average_tokens_per_second": (
            total_tokens_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "latest_checkpoint": str(checkpoint_path(args.checkpoint_dir, "latest")),
        "best_checkpoint": str(checkpoint_path(args.checkpoint_dir, "best")),
        "warnings": warnings,
    }
    write_json(Path(args.checkpoint_dir) / "final_summary.json", final_summary)
    if aim_run is not None:
        aim_run["final_summary"] = final_summary
    print(json.dumps(final_summary, indent=2))
    print(f"Finished training for {args.epochs} epoch(s)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyGPT")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tiny_gpt/tokenizer.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/tiny_gpt")
    parser.add_argument("--dataset", default="skeskinen/TinyStories-hf")
    parser.add_argument("--dataset_config")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--validation_split", default="validation")
    parser.add_argument("--text_column", default="text")

    parser.add_argument("--chunk_size", type=int, default=25000)

    parser.add_argument("--context_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--lr_decay_steps", type=int)
    parser.add_argument("--min_lr", type=float, default=1e-4)

    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--aim_repo", default="runs/aim")
    parser.add_argument("--aim_experiment", default="tiny-gpt")
    parser.add_argument("--disable_aim", action="store_true")

    parser.add_argument("--keep_step_checkpoints", type=int, default=3)
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--disable_sample_logging", action="store_true")
    parser.add_argument("--sample_prompt", action="append")
    parser.add_argument("--sample_max_new_tokens", type=int, default=96)
    parser.add_argument("--sample_temperature", type=float, default=0.8)
    parser.add_argument("--sample_top_k", type=int, default=40)
    parser.add_argument("--sample_seed", type=int, default=1337)
    parser.add_argument("--samples_path", default="runs/training_samples.jsonl")

    args = parser.parse_args()
    main(args)
