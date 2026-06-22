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
    def __init__(self, model, train_loader, val_loader, config: TrainerConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self):
        train_losses = []
        val_losses = []
        val_steps = []
        tokens_per_sec_history = []

        self.model.train()

        for step, (x, y) in enumerate(self.train_loader):
            start = time.time()
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)

            loss = self.loss_fn(
                logits.view(-1, self.model.vocab_size),
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

            if step % self.config.eval_interval == 0:
                vl = self.estimate_loss()
                val_losses.append(vl)
                val_steps.append(step)

                print(
                    "step", step,
                    "train loss", loss.item(),
                    "val loss", vl,
                )

        return val_steps, train_losses, val_losses, tokens_per_sec_history

    def estimate_loss(self):
        self.model.eval()

        losses = []

        with torch.no_grad():
            for step, (x, y) in enumerate(self.val_loader):
                if step == self.config.eval_batches:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                loss = self.loss_fn(
                    logits.view(-1, self.model.vocab_size),
                    y.view(-1)
                )
                losses.append(loss.item())

        self.model.train()

        return statistics.mean(losses)


