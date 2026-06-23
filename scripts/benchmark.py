import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_gpt.configuration_tiny_gpt import TinyGPTConfig
from tiny_gpt.modeling_tiny_gpt import TinyGPTForCausalLM


def parse_batch_sizes(value):
    return [int(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark complete Tiny GPT optimizer steps."
    )
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[8, 16, 32],
        help="Comma-separated microbatch sizes.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=128,
        help="Sequences per optimizer step. Must divide by each batch size.",
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--accumulation", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_model(args, device):
    config = TinyGPTConfig(
        context_size=args.context_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=0.0,
    )
    return TinyGPTForCausalLM(config).to(device).train()


def optimizer_step(
    model,
    optimizer,
    input_ids,
    labels,
    accumulation,
    grad_clip,
    device,
):
    final_loss = None
    for _ in range(accumulation):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(input_ids=input_ids).logits
            final_loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
            )
        (final_loss / accumulation).backward()

    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return final_loss


def run_worker(args):
    if args.batch_size is None or args.accumulation is None:
        raise ValueError("Worker requires batch size and accumulation")

    torch.manual_seed(args.seed + args.repetition)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.repetition)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    optimizer.zero_grad(set_to_none=True)

    input_ids = torch.randint(
        model.config.vocab_size,
        (args.batch_size, model.config.context_size),
        device=device,
    )
    labels = torch.randint(
        model.config.vocab_size,
        input_ids.shape,
        device=device,
    )

    for _ in range(args.warmup_steps):
        optimizer_step(
            model,
            optimizer,
            input_ids,
            labels,
            args.accumulation,
            args.grad_clip,
            device,
        )

    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    final_loss = None
    for _ in range(args.steps):
        final_loss = optimizer_step(
            model,
            optimizer,
            input_ids,
            labels,
            args.accumulation,
            args.grad_clip,
            device,
        )
    synchronize()
    elapsed = time.perf_counter() - start

    tokens = (
        args.steps
        * args.batch_size
        * args.accumulation
        * model.config.context_size
    )
    report = {
        "batch_size": args.batch_size,
        "gradient_accumulation": args.accumulation,
        "effective_batch_size": args.batch_size * args.accumulation,
        "repetition": args.repetition,
        "optimizer_steps": args.steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "milliseconds_per_optimizer_step": elapsed * 1000 / args.steps,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else None
        ),
        "final_loss": final_loss.item(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "precision": "bfloat16" if device.type == "cuda" else "float32",
    }
    print(json.dumps(report))


def worker_command(args, batch_size, accumulation, repetition):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--context-size",
        str(args.context_size),
        "--vocab-size",
        str(args.vocab_size),
        "--d-model",
        str(args.d_model),
        "--n-layers",
        str(args.n_layers),
        "--n-heads",
        str(args.n_heads),
        "--batch-size",
        str(batch_size),
        "--accumulation",
        str(accumulation),
        "--warmup-steps",
        str(args.warmup_steps),
        "--steps",
        str(args.steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--seed",
        str(args.seed),
        "--repetition",
        str(repetition),
    ]
    return command


def summarize(runs):
    throughputs = [run["tokens_per_second"] for run in runs]
    step_times = [run["milliseconds_per_optimizer_step"] for run in runs]
    memories = [run["peak_memory_bytes"] for run in runs]
    return {
        "batch_size": runs[0]["batch_size"],
        "gradient_accumulation": runs[0]["gradient_accumulation"],
        "effective_batch_size": runs[0]["effective_batch_size"],
        "median_tokens_per_second": statistics.median(throughputs),
        "mean_tokens_per_second": statistics.mean(throughputs),
        "stdev_tokens_per_second": (
            statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        ),
        "min_tokens_per_second": min(throughputs),
        "max_tokens_per_second": max(throughputs),
        "median_milliseconds_per_optimizer_step": statistics.median(step_times),
        "peak_memory_bytes": max(memories),
        "runs": runs,
    }


def run_controller(args):
    for batch_size in args.batch_sizes:
        if args.effective_batch_size % batch_size:
            raise ValueError(
                f"effective_batch_size={args.effective_batch_size} is not "
                f"divisible by batch_size={batch_size}"
            )

    summaries = []
    for batch_size in args.batch_sizes:
        accumulation = args.effective_batch_size // batch_size
        runs = []
        for repetition in range(args.repetitions):
            print(
                f"Benchmarking batch={batch_size}, "
                f"accumulation={accumulation}, "
                f"repetition={repetition + 1}/{args.repetitions}...",
                flush=True,
            )
            result = subprocess.run(
                worker_command(
                    args,
                    batch_size,
                    accumulation,
                    repetition,
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            runs.append(json.loads(result.stdout.strip().splitlines()[-1]))
        summaries.append(summarize(runs))

    report = {
        "model": {
            "context_size": args.context_size,
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
        },
        "methodology": {
            "warmup_optimizer_steps": args.warmup_steps,
            "measured_optimizer_steps": args.steps,
            "repetitions": args.repetitions,
            "effective_batch_size": args.effective_batch_size,
            "isolated_process_per_run": True,
            "includes": [
                "forward",
                "backward",
                "gradient_clipping",
                "AdamW",
                "gradient_accumulation",
            ],
        },
        "results": summaries,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
