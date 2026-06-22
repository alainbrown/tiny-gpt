import statistics
import time
import os
import torch

from dataclasses import dataclass

@dataclass
class TrainerConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    eval_interval: int = 1000
    eval_batches: int = 100
    checkpoint_dir: str = "checkpoints/tiny_gpt"

class Trainer:
    def __init__(self, model, config: TrainerConfig):
        self.model = model
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train_steps(self, train_iter, num_steps):
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

            logits = self.model(x)

            loss = self.loss_fn(
                logits.view(-1, self.model.config.vocab_size),
                y.view(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())

            elapsed = time.time() - start
            tokens_processed = x.numel()
            tokens_per_second = tokens_processed / elapsed
            tokens_per_sec_history.append(tokens_per_second)

            # Print occasionally to show it's alive
            if step % max(1, self.config.eval_interval // 10) == 0:
                print(f"   [Step {step}/{num_steps}] train loss {loss.item():.4f} ({tokens_per_second:.0f} tok/s)")

        return train_losses, tokens_per_sec_history

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

                logits = self.model(x)

                loss = self.loss_fn(
                    logits.view(-1, self.model.config.vocab_size),
                    y.view(-1)
                )
                losses.append(loss.item())

        self.model.train()
        return statistics.mean(losses) if losses else 0.0
