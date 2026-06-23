import contextlib
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch


@dataclass
class TrainStepStats:
    losses: list[float]
    tokens_per_sec_history: list[float]
    global_step: int
    grad_norm_history: list[float]
    clipped_steps: int
    optimizer_steps: int
    tokens_processed: int

    def __iter__(self):
        yield self.losses
        yield self.tokens_per_sec_history
        yield self.global_step


@dataclass
class TrainerConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    lr_decay_steps: Optional[int] = None
    min_lr: float = 1e-4
    eval_interval: int = 1000
    eval_batches: int = 100
    checkpoint_dir: str = "checkpoints/tiny_gpt"

class Trainer:
    """
    A pure mathematical step-processor for training Language Models.

    Techniques implemented:
    - Mixed Precision Training (bfloat16): Accelerates forward/backward passes while halving VRAM usage.
    - Gradient Accumulation: Simulates massive batch sizes without running out of VRAM.
    - Cosine LR Decay with Warmup: Ramps up LR slowly to stabilize initialization, then gracefully arcs down.
    - Gradient Clipping: Hard-caps explosive gradients to prevent training collapse.
    - AdamW Optimizer with Weight Decay: Penalizes massive weights to prevent overfitting.
    """
    def __init__(self, model, config: TrainerConfig):
        self.model = model
        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer.zero_grad(set_to_none=True)

    def get_lr(self, it):
        if it < self.config.warmup_steps:
            return self.config.learning_rate * (it + 1) / self.config.warmup_steps

        if self.config.lr_decay_steps is None:
            return self.config.learning_rate

        lr_decay_steps = max(self.config.lr_decay_steps, self.config.warmup_steps + 1)
        if it > lr_decay_steps:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_steps) / (lr_decay_steps - self.config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def reached_max_steps(self, global_step):
        return (
            self.config.max_steps is not None
            and global_step >= self.config.max_steps
        )

    def autocast_context(self):
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def optimizer_step(self, accumulated_steps):
        if accumulated_steps < self.config.grad_accum_steps:
            correction = self.config.grad_accum_steps / accumulated_steps
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)

        grad_norm = torch.tensor(0.0, device=self.device)
        if self.config.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )
        else:
            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            if parameters:
                grad_norm = torch.norm(
                    torch.stack(
                        [
                            torch.norm(parameter.grad.detach(), 2)
                            for parameter in parameters
                        ]
                    ),
                    2,
                )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.detach().cpu())

    def save_state(self, path, global_step):
        os.makedirs(path, exist_ok=True)
        state = {
            "optimizer": self.optimizer.state_dict(),
            "global_step": global_step,
            "trainer_config": asdict(self.config),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(state, os.path.join(path, "trainer_state.pt"))

    def load_optimizer_state(self, state):
        self.optimizer.load_state_dict(state)
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)

    def load_state(self, path):
        state_path = os.path.join(path, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
            self.load_optimizer_state(state["optimizer"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if "cuda_rng_state" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            return state.get("global_step", 0)

        legacy_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(legacy_path):
            state = torch.load(
                legacy_path,
                map_location="cpu",
                weights_only=True,
            )
            self.load_optimizer_state(state)
        return None

    def train_steps(self, train_iter, num_steps, global_step=0):
        train_losses = []
        tokens_per_sec_history = []
        grad_norm_history = []
        clipped_steps = 0
        optimizer_steps = 0
        tokens_processed = 0
        accumulated_steps = 0
        log_interval = max(1, self.config.eval_interval // 10)
        window_tokens = 0
        window_steps = 0

        self.model.train()
        self.synchronize()
        window_start = time.perf_counter()

        for step in range(num_steps):
            if self.reached_max_steps(global_step):
                break

            try:
                x, y = next(train_iter)
            except StopIteration:
                break

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            lr = self.get_lr(global_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            with self.autocast_context():
                logits = self.model(x).logits
                loss = self.loss_fn(
                    logits.view(-1, self.model.config.vocab_size),
                    y.view(-1),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at global step {global_step}: "
                    f"{loss.item()}"
                )

            (loss / self.config.grad_accum_steps).backward()
            accumulated_steps += 1

            if accumulated_steps == self.config.grad_accum_steps:
                grad_norm = self.optimizer_step(accumulated_steps)
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at global step {global_step}: "
                        f"{grad_norm}"
                    )
                grad_norm_history.append(grad_norm)
                if (
                    self.config.grad_clip != 0.0
                    and grad_norm > self.config.grad_clip
                ):
                    clipped_steps += 1
                optimizer_steps += 1
                accumulated_steps = 0
                global_step += 1

            train_losses.append(loss.item())
            window_tokens += x.numel()
            tokens_processed += x.numel()
            window_steps += 1

            if (
                window_steps == log_interval
                or self.reached_max_steps(global_step)
            ):
                self.synchronize()
                elapsed = time.perf_counter() - window_start
                tokens_per_second = window_tokens / elapsed
                tokens_per_sec_history.append(tokens_per_second)
                print(
                    f"   [Step {step}/{num_steps}] "
                    f"train loss {loss.item():.4f} "
                    f"({tokens_per_second:.0f} tok/s) | lr: {lr:.2e}"
                )
                window_tokens = 0
                window_steps = 0
                window_start = time.perf_counter()

        if accumulated_steps and not self.reached_max_steps(global_step):
            grad_norm = self.optimizer_step(accumulated_steps)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at global step {global_step}: "
                    f"{grad_norm}"
                )
            grad_norm_history.append(grad_norm)
            if self.config.grad_clip != 0.0 and grad_norm > self.config.grad_clip:
                clipped_steps += 1
            optimizer_steps += 1
            global_step += 1

        if window_steps:
            self.synchronize()
            elapsed = time.perf_counter() - window_start
            tokens_per_sec_history.append(window_tokens / elapsed)

        return TrainStepStats(
            losses=train_losses,
            tokens_per_sec_history=tokens_per_sec_history,
            global_step=global_step,
            grad_norm_history=grad_norm_history,
            clipped_steps=clipped_steps,
            optimizer_steps=optimizer_steps,
            tokens_processed=tokens_processed,
        )

    def estimate_loss(self, val_iter):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for step in range(self.config.eval_batches):
                try:
                    x, y = next(val_iter)
                except StopIteration:
                    break

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with self.autocast_context():
                    logits = self.model(x).logits
                    loss = self.loss_fn(
                        logits.view(-1, self.model.config.vocab_size),
                        y.view(-1),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite validation loss at validation step "
                        f"{step}: {loss.item()}"
                    )
                losses.append(loss.item())

        self.model.train()
        return statistics.mean(losses) if losses else 0.0
