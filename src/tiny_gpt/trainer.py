import statistics
import time
import os
import torch

from dataclasses import dataclass

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
    max_steps: int = 100000
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def get_lr(self, it):
        import math
        if it < self.config.warmup_steps:
            return self.config.learning_rate * (it + 1) / self.config.warmup_steps
        if it > self.config.max_steps:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def train_steps(self, train_iter, num_steps, global_step=0):
        train_losses = []
        tokens_per_sec_history = []

        self.model.train()

        for step in range(num_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                break

            start = time.time()
            x = x.to(self.device)
            y = y.to(self.device)

            lr = self.get_lr(global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(x).logits
                loss = self.loss_fn(
                    logits.view(-1, self.model.config.vocab_size),
                    y.view(-1)
                )
                loss = loss / self.config.grad_accum_steps

            loss.backward()
            
            if (step + 1) % self.config.grad_accum_steps == 0 or (step + 1) == num_steps:
                if self.config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1

            train_losses.append(loss.item() * self.config.grad_accum_steps)

            elapsed = time.time() - start
            tokens_processed = x.numel()
            tokens_per_second = tokens_processed / elapsed
            tokens_per_sec_history.append(tokens_per_second)

            # Print occasionally to show it's alive
            if step % max(1, self.config.eval_interval // 10) == 0:
                print(f"   [Step {step}/{num_steps}] train loss {loss.item() * self.config.grad_accum_steps:.4f} ({tokens_per_second:.0f} tok/s) | lr: {lr:.2e}")

        return train_losses, tokens_per_sec_history, global_step

    def estimate_loss(self, val_iter):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for step in range(self.config.eval_batches):
                try:
                    x, y = next(val_iter)
                except StopIteration:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.model(x).logits
                    loss = self.loss_fn(
                        logits.view(-1, self.model.config.vocab_size),
                        y.view(-1)
                    )
                losses.append(loss.item())

        self.model.train()
        return statistics.mean(losses) if losses else 0.0
