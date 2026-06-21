import math
import torch
from torch import nn

"""
token embeddings
learned positional embeddings
causal self-attention
feed-forward network
custom LayerNorm
residual connections
stacked transformer blocks
GELU
dropout
Pre-LayerNorm
tied token embedding/output projection weights
multi-head attention

"""
class Model(nn.Module):
    def __init__(self, context_size, vocab_size, d_model, n_layers, n_heads, dropout=0.1):
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
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
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
        variance = (diff * diff).mean(dim=-1, keepdim=True)
        normalized = diff / torch.sqrt(variance + 1e-6)
        return self.gamma * normalized + self.beta

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // self.n_heads 
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        attention = self.self_attention(self.layer_norm1(x))
        attention = self.dropout(attention)

        x = x + attention

        feed_forward = self.feed_forward(self.layer_norm2(x))
        feed_forward = self.dropout(feed_forward)

        x = x + feed_forward

        return x

    def self_attention(self, x):
        query = self.split_heads(self.query(x))
        key = self.split_heads(self.key(x))
        value = self.split_heads(self.value(x))

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / self.scale

        context_size = query.shape[2]
        mask = torch.tril(torch.ones(context_size, context_size, device=query.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.nn.functional.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = self.combine_heads(attended)
        attended = self.head_proj(attended)

        return attended

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape

        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        batch_size, n_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, -1)

        return x
