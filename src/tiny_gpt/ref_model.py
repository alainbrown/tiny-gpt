import math

import torch
from torch import nn


class ReferenceGPTModel(nn.Module):
    """Explicit decoder-only Transformer used as a mathematical reference."""

    def __init__(
        self,
        context_size,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        dropout=0.1,
    ):
        super().__init__()

        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                ReferenceTransformerBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.final_layer_norm = ReferenceLayerNorm(d_model)

    def forward(self, x):
        _, sequence_length = x.shape

        assert sequence_length <= self.context_size, (
            "Input sequence is longer than context_size"
        )

        positions = torch.arange(sequence_length, device=x.device)
        position = self.position_embedding(positions)
        token = self.token_embedding(x)

        x = self.dropout(token + position)
        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_layer_norm(x)
        return self.linear(x)


class ReferenceFeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        x = self.ff1(x)
        x = nn.functional.gelu(x)
        return self.ff2(x)


class ReferenceLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        variance = (diff * diff).mean(dim=-1, keepdim=True)
        normalized = diff / torch.sqrt(variance + 1e-6)
        return self.gamma * normalized + self.beta


class ReferenceMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0, (
            "d_model must be divisible by n_heads"
        )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.head_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        query = self.split_heads(self.query(x))
        key = self.split_heads(self.key(x))
        value = self.split_heads(self.value(x))

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / self.scale

        context_size = query.shape[2]
        mask = torch.tril(
            torch.ones(context_size, context_size, device=query.device)
        )
        mask = mask.view(1, 1, context_size, context_size)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.nn.functional.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        return self.head_proj(self.combine_heads(attended))

    def split_heads(self, x):
        batch_size, sequence_length, _ = x.shape
        x = x.reshape(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, n_heads, sequence_length, head_dim = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(
            batch_size,
            sequence_length,
            n_heads * head_dim,
        )


class ReferenceTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.feed_forward = ReferenceFeedForward(d_model)
        self.layer_norm1 = ReferenceLayerNorm(d_model)
        self.layer_norm2 = ReferenceLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.multi_head_attention = ReferenceMultiHeadAttention(
            d_model,
            n_heads,
            dropout,
        )

    def forward(self, x):
        attention = self.multi_head_attention(self.layer_norm1(x))
        x = x + self.dropout(attention)

        feed_forward = self.feed_forward(self.layer_norm2(x))
        return x + self.dropout(feed_forward)
