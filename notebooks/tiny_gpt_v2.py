# -*- coding: utf-8 -*-
"""tiny_gpt_v2.ipynb

# Dataset Loader
"""

from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

import torch

class Dataset:

  def __init__(self, tokens, context):
    self.tokens = tokens
    self.context = context

  def __getitem__(self, index):
      x = self.tokens[index : index + self.context]
      y = self.tokens[index + 1 : index + self.context + 1]

      return (
          torch.tensor(x, dtype=torch.long),
          torch.tensor(y, dtype=torch.long),
      )

  def __len__(self):
      return len(self.tokens) - self.context

from torch.utils.data import DataLoader

def create_dataloaders(ds, context, batch_size, n_stories):

    encoding = []
    for story in ds["train"][:n_stories]["text"]:
      encoding.extend(tokenizer.encode(story, add_eos=True))

    train_ds = Dataset(encoding, context=context)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    encoding = []
    for story in ds["validation"][:n_stories]["text"]:
      encoding.extend(tokenizer.encode(story, add_eos=True))

    val_ds = Dataset(encoding, context=context)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

"""# BPE Tokenizer"""

from collections import Counter
import json


class BPETokenizer:
    def __init__(self):
        # Base vocabulary: every possible byte.
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Special tokens live outside byte range.
        self.eos_id = 256
        self.vocab[self.eos_id] = b""

        # Dictionary mapping pair to new_id:
        # {(left_id, right_id): new_id}
        self.merges = {}

    def text_to_ids(self, text):
        return list(text.encode("utf-8"))

    def get_pair_counts(self, sequences):
        counts = Counter()

        for ids in sequences:
            for pair in zip(ids, ids[1:]):
                counts[pair] += 1

        return counts

    def apply_merge(self, ids, pair, new_id):
        out = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1

        return out

    def train(self, texts, vocab_size):
        assert vocab_size > 257

        # Tokenizer training corpus.
        # This can be a subset of the full dataset.
        sequences = [self.text_to_ids(text) for text in texts]

        next_id = 257

        while next_id < vocab_size:
            counts = self.get_pair_counts(sequences)

            if len(counts) == 0:
                break

            pair, count = counts.most_common(1)[0]

            if count < 2:
                break

            new_id = next_id
            next_id += 1

            self.merges[pair] = new_id

            left, right = pair
            self.vocab[new_id] = self.vocab[left] + self.vocab[right]

            sequences = [
                self.apply_merge(ids, pair, new_id)
                for ids in sequences
            ]

    def encode(self, text, add_eos=False):
        ids = self.text_to_ids(text)

        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break
                
            new_id = self.merges[pair]
            ids = self.apply_merge(ids, pair, new_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, skip_special=True):
        chunks = []

        for idx in ids:
            if idx == self.eos_id and skip_special:
                continue

            chunks.append(self.vocab[idx])

        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "eos_id": self.eos_id,
            "merges": [
                [left, right, new_id]
                for (left, right), new_id in self.merges.items()
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        tokenizer = cls()

        with open(path, "r") as f:
            data = json.load(f)

        tokenizer.eos_id = data["eos_id"]
        tokenizer.merges = {}

        for left, right, new_id in data["merges"]:
            pair = (left, right)
            tokenizer.merges[pair] = new_id
            tokenizer.vocab[new_id] = (
                tokenizer.vocab[left] + tokenizer.vocab[right]
            )

        return tokenizer

from google.colab import drive
import pathlib

drive.mount("/content/drive")
path = "/content/drive/MyDrive/tiny_bpe_v2.pt"

tokenizer = BPETokenizer()

if pathlib.Path(path).exists():
    tokenizer.load(path)
else:
    tokenizer_texts = ds["train"][:1000]["text"]
    tokenizer.train(tokenizer_texts, vocab_size=1024)

tokens = tokenizer.encode("The dragon flew over the castle.")
print(tokens)
print(tokenizer.decode(tokens))
print()

for sentence in [
    "Hello world",
    "The dragon flew over the castle.",
    "banana banana banana"
]:
    ids = tokenizer.encode(sentence)
    decoded = tokenizer.decode(ids)
    print(decoded == sentence)

"""# Transformer"""

import math
import torch
from torch import nn

class GPT(nn.Module):
    def __init__(self, context_size, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()

        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, dropout) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.linear.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)

        position = self.position_embedding(positions)
        token = self.token_embedding(x)

        x = token + position
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.linear(x)

        return logits

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x = self.ff1(x)
        x = nn.functional.gelu(x)
        x = self.ff2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        diff = (x - mean)
        variance = (diff*diff).mean(dim=-1, keepdim=True)
        normalized = diff / torch.sqrt(variance + 1e-6)
        return self.gamma * normalized + self.beta

class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.self_attention(self.layer_norm1(x))
        attention = self.dropout(attention)

        x = x + attention

        feed_forward = self.feed_forward(self.layer_norm2(x))
        feed_forward = self.dropout(feed_forward)

        x = x + feed_forward

        return x

    def self_attention(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / self.scale

        context_size = query.shape[1]
        mask = torch.tril(torch.ones(context_size, context_size, device=query.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.nn.functional.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        return attended

"""# Training"""

import statistics
import time

class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, eval_interval, eval_batches):
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

            if step % eval_interval == 0:
                vl = self.estimate_loss(eval_batches)
                val_losses.append(vl)
                val_steps.append(step)

                print(
                    "step", step,
                    "train loss", loss.item(),
                    "val loss", vl,
                )

        return val_steps, train_losses, val_losses, tokens_per_sec_history

    def estimate_loss(self, eval_batches):
        self.model.eval()

        losses = []

        with torch.no_grad():
            for step, (x, y) in enumerate(self.val_loader):
                if step == eval_batches:
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

    def save_checkpoint(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config = checkpoint["config"]

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


dims = 64
context = 32
vocab_size = len(tokenizer.vocab)
batch_size = 16
learning_rate = 1e-3
n_layers = 4

model = GPT(
    context_size=context,
    vocab_size=vocab_size,
    d_model=dims,
    n_layers=n_layers,
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_loader, val_loader = create_dataloaders(ds, context, batch_size, 1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

config = {
    "context_size": context,
    "vocab_size": vocab_size,
    "d_model": dims,
    "n_layers": n_layers,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
}

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=config,
)

val_steps, train_losses, val_losses, tokens_per_sec_history = trainer.train(
    eval_interval=1000,
    eval_batches=100,
)

def moving_average(xs, window=100):
    return [
        sum(xs[i:i+window]) / len(xs[i:i+window])
        for i in range(len(xs) - window)
    ]

smoothed = moving_average(train_losses, window=100)

plt.plot(smoothed, label="ma training loss")
plt.plot(train_losses, label="train")
plt.plot(val_steps, val_losses, label="val")
plt.loglog()
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Training loss")
plt.legend()
plt.show()

plt.plot(tokens_per_sec_history)
plt.xlabel("step")
plt.ylabel("tokens/sec")
plt.title("Training throughput")
plt.show()

"""# Checkpoints"""

from google.colab import drive
drive.mount("/content/drive")

def save_checkpoint(path, model, optimizer, config):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["config"]

path = "/content/drive/MyDrive/tiny_gpt_v2.pt"

config = {
    "context": context,
    "vocab_size": vocab_size,
    "d_model": dims,
    "n_layers": 4,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
}

save_checkpoint(path, model, optimizer, config)
load_checkpoint(path, model, optimizer)

"""# Eval

## Eval Prompts
"""

eval_prompts = [
    "",
    "The",
    "Once upon a time",
    "There was",
    "The dragon",
    "A little girl",
]

def run_eval_prompts(model, tokenizer, context_size, max_new_tokens=100):
    for prompt in eval_prompts:
        prompt_ids = tokenizer.encode(prompt)

        out = generate(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            tokenizer=tokenizer,
        )

        print("=" * 80)
        print("PROMPT:", repr(prompt))
        print(tokenizer.decode(out))

"""## Generation"""

import torch

def generate(
    model,
    prompt_ids,
    max_new_tokens,
    context_size,
    tokenizer,
    mode="sample",
    temperature=1.0,
    top_k=None,
):
    model.eval()

    device = next(model.parameters()).device

    x = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    eos_id = tokenizer.vocab.get("<EOS>")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -context_size:]

            logits = model(x_cond)

            logits = logits[:, -1, :]

            if mode == "greedy":
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            elif mode == "sample":
                logits = logits / temperature

                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)

                    filtered_logits = torch.full_like(logits, float("-inf"))
                    filtered_logits.scatter_(dim=-1, index=indices, src=values)

                    logits = filtered_logits

                probs = torch.softmax(logits, dim=-1)

                next_id = torch.multinomial(probs, num_samples=1)

            else:
                raise ValueError(f"Unknown generation mode: {mode}")

            x = torch.cat([x, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

    return x.squeeze(0).tolist()

prompt = tokenizer.encode("Once upon a time")

settings = [
    {
        "mode": "greedy",
    },
    {
        "mode": "sample",
        "temperature": 1.0,
        "top_k": None,
    },
    {
        "mode": "sample",
        "temperature": 0.8,
        "top_k": 20,
    },
    {
        "mode": "sample",
        "temperature": 1.2,
        "top_k": 40,
    },
]

for setting in settings:
    out = generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=100,
        context_size=context,
        tokenizer=tokenizer,
        **setting,
    )

    print("=" * 80)
    print(setting)
    print(tokenizer.decode(out))

len(train_loader.dataset)
len(train_loader)