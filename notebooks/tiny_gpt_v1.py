# -*- coding: utf-8 -*-
"""tiny_gpt_v1.ipynb

# Dataset Loader
"""

from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

print(ds["train"][0])

"""# Tokenizer

## Trivial Tokenizer
"""

text = ds["train"][0]["text"]

chars = sorted(list(set(ds["train"][0]["text"])))

atoi = {ch: i for i, ch in enumerate(chars)}
itoa = {i: ch for i, ch in enumerate(chars)}

def encode(text):
  return [atoi[ch] for ch in text]

def decode(tokens):
  return "".join([itoa[token] for token in tokens])

decode(encode(text)) == text

"""## BPE Tokenizer"""

class Tokenizer:
  def __init__(self, corpus, N):
    self.merges, self.vocab = bpe(corpus, N)
    self.reverse_vocab = {v: k for k, v in self.vocab.items()}

  def encode(self, text):
    tokens = list(text)

    for pair in self.merges:
      tokens = apply_merge(tokens, pair)

    return [self.vocab[t] for t in tokens]

  def decode(self, tokens):

    decoding = [self.reverse_vocab[t] for t in tokens]

    return "".join(decoding)

def pair_count(tokens):
  counts = {}

  for i in range(len(tokens) - 1):
    pair = (tokens[i], tokens[i + 1])
    counts[pair] = counts.get(pair, 0) + 1

  return counts

def get_top_pair(pair_count):
  best_pair = None
  best_count = 0

  for pair, count in pair_count.items():
    if count > best_count:
      best_pair = pair
      best_count = count

  return best_pair

def apply_merge(tokens, pair):
  merged_tokens = []
  i = 0

  while i < len(tokens):
    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
      merged_tokens.append(pair[0] + pair[1])
      i += 1
    else:
      merged_tokens.append(tokens[i])

    i += 1

  return merged_tokens

def bpe(text, N):
  tokens = text
  vocab = {}
  index = 0
  merges = set([])

  for ch in tokens:
    if ch not in vocab:
      vocab[ch] = index
      index += 1

  while len(vocab) < N:
    counts = pair_count(tokens)
    pair = get_top_pair(counts)

    if pair == None or counts[pair] < 2:
      break

    token = pair[0] + pair[1]

    if token not in vocab:
      vocab[token] = index
      index += 1

    merges.add(pair)
    tokens = apply_merge(tokens, pair)

  return merges, vocab

"""## Train Tokenizer on TinyStories"""

tokens = []
for story in ds["train"][:100]["text"]:
  tokens.extend(list(story))
  tokens.append("<EOS>")

tokenizer = Tokenizer(tokens, 128)
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

"""## Dataset token stream"""

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

encoding = []

for story in ds["train"][:100]["text"]:
  ids = tokenizer.encode(story)
  encoding.extend(ids)
  encoding.append(tokenizer.vocab["<EOS>"])

dataset = Dataset(encoding, context=8)

x, y = dataset[0]

print(x)
print(y)

assert len(x) == 8
assert len(y) == 8

# Every target token should be the next input token
assert torch.equal(x[1:], y[:-1])

"""# Transformer

## Tiny transformer and batch generation
"""

import torch
from torch import nn

class GPT(torch.nn.Module):
    def __init__(self, context_size, vocab_size, d_model):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.linear = nn.Linear(vocab_size, d_model)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        position = self.position_embedding(positions)
        x = self.token_embedding(x) + position
        y = self.linear(x)
        return y

dataset = Dataset(encoding, context=32)
vocab_size = len(tokenizer.vocab)
gpt = GPT(32, vocab_size, 128)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)

for step in range(1000):
    x, y = dataset[step]

    x = x.unsqueeze(0)  # batch dimension
    y = y.unsqueeze(0)

    logits = gpt(x)

    loss = loss_fn(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())

import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [1, 2, 3]])

print(torch.matmul(x, torch.tensor([1, 2, 3])))
print(x / 2)
print(x * 2)
mask = torch.tril(torch.ones(2, 3))
print(x.masked_fill(mask == 0, 0))

"""## Tramsformer"""

import math
import torch
from torch import nn

class GPT(nn.Module):
    def __init__(self, context_size, vocab_size, d_model, n_layers):
        super().__init__()

        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        position = self.position_embedding(positions)
        x = self.token_embedding(x) + position

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
        x = torch.relu(x)
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
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x):
        attention = self.self_attention(x)
        x = x + attention
        x = self.layer_norm1(x)

        feed_forward = self.feed_forward(x)
        x = x + feed_forward
        x = self.layer_norm2(x)
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

"""Sanity Check"""

context = 32
vocab = 1000
dims = 64

model = GPT(
    context_size=context,
    vocab_size=vocab,
    d_model=dims,
    n_layers=4,
)

print(model)
total = sum(p.numel() for p in model.parameters())
print(total)

x = torch.randint(0, vocab, (2, 32))
model(x)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
x = torch.randint(0, vocab, (2, 32))
y = torch.randint(0, vocab, (2, 32))

logits = model(x)

loss = loss_fn(
    logits.view(-1, vocab),
    y.view(-1)
)

print(model.linear.weight.grad)

loss.backward()

print(model.linear.weight.grad)

print(model.linear.weight)
optimizer.step()
print(model.linear.weight)

"""# Training

## Small Training Loop
"""

from torch.utils.data import DataLoader

tokens = []
for story in ds["train"][:100]["text"]:
  tokens.extend(list(story))
  tokens.append("<EOS>")

tokenizer = Tokenizer(tokens, 128)

encoding = []
for story in ds["train"][:100]["text"]:
  ids = tokenizer.encode(story)
  encoding.extend(ids)
  encoding.append(tokenizer.vocab["<EOS>"])

dims = 64
context = 32
vocab_size = len(tokenizer.vocab)
batch_size = 16
learning_rate = 1e-3

model = GPT(
    context_size=context,
    vocab_size=vocab_size,
    d_model=dims,
    n_layers=4,
)

dataset = Dataset(encoding, context=context)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for step, (x,y) in enumerate(loader):
    logits = model(x)

    loss = loss_fn(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())

"""## Generation Test"""

import torch

def generate(model, prompt_ids, max_new_tokens, context_size, tokenizer):
    model.eval()

    device = next(model.parameters()).device
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    eos_id = tokenizer.vocab.get("<EOS>")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -context_size:]

            logits = model(x_cond)
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, 1)

            x = torch.cat([x, next_id], dim=1)

            if next_id.item() == eos_id:
                break

    return x.squeeze(0).tolist()

"""Sanity Check Generation"""

prompt = tokenizer.encode("Once upon a time")
prompts = [
    "The",
    "Once upon a time",
    "There was",
    "The dragon",
    "A little girl",
]
for prompt in prompts:
  prompt_ids = tokenizer.encode(prompt)
  out = generate(
      model=model,
      prompt_ids=prompt_ids,
      max_new_tokens=100,
      context_size=context,
      tokenizer=tokenizer
  )
  print("=" * 60)
  print(prompt)
  print("-" * 60)
  print(tokenizer.decode(out))

"""## Overfitting Diagnostic
Train loss and validation loss
"""

from torch.utils.data import DataLoader
import statistics
import torch

volume = 100

encoding = []
for story in ds["train"][:volume]["text"]:
    ids = tokenizer.encode(story)
    encoding.extend(ids)
    encoding.append(tokenizer.vocab["<EOS>"])

dims = 64
context = 32
vocab_size = len(tokenizer.vocab)
batch_size = 16
learning_rate = 1e-3

model = GPT(
    context_size=context,
    vocab_size=vocab_size,
    d_model=dims,
    n_layers=4,
)

split = int(0.9 * len(encoding))

train_tokens = encoding[:split]
val_tokens = encoding[split:]

train_dataset = Dataset(train_tokens, context=context)
val_dataset = Dataset(val_tokens, context=context)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

def estimate_loss(model, loader, n_loss_batches):
    model.eval()

    losses = []

    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            if step == n_loss_batches:
                break

            logits = model(x)

            loss = loss_fn(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            losses.append(loss.item())

    model.train()

    return statistics.mean(losses)


def train(model, train_loader, val_loader, vocab_size, n_loss_batches):
    train_losses = []
    val_losses = []

    model.train()

    for step, (x, y) in enumerate(train_loader):
        logits = model(x)

        loss = loss_fn(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if step % n_loss_batches == 0:
            vl = estimate_loss(model, val_loader, n_loss_batches)
            val_losses.append(vl)

            print(
                "step", step,
                "train loss", loss.item(),
                "val loss", vl,
            )

    return val_losses, train_losses


val_loss, train_loss = train(
    model,
    train_loader,
    val_loader,
    vocab_size,
    n_loss_batches=100,
)

print("mean val loss:", statistics.mean(val_loss))
print("mean train loss:", statistics.mean(train_loss))

"""## Trainer"""

class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config

    def train(self, n_loss_batches):
        train_losses = []
        val_losses = []

        self.model.train()

        for step, (x, y) in enumerate(self.train_loader):
            logits = self.model(x)

            loss = self.loss_fn(
                logits.view(-1, self.model.vocab_size),
                y.view(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())

            if step % n_loss_batches == 0:
                vl = self.estimate_loss(n_loss_batches)
                val_losses.append(vl)

                print(
                    "step", step,
                    "train loss", loss.item(),
                    "val loss", vl,
                )

        return val_losses, train_losses

    def estimate_loss(self, n_loss_batches):
        self.model.eval()

        losses = []

        with torch.no_grad():
            for step, (x, y) in enumerate(self.val_loader):
                if step == n_loss_batches:
                    break

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

path = "/content/drive/MyDrive/tiny_gpt.pt"

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

prompt = tokenizer.encode("Once upon a time")

out = generate(
    model,
    prompt,
    max_new_tokens=100,
    context_size=config["context"],
    tokenizer=tokenizer,
)

print(tokenizer.decode(out))

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

