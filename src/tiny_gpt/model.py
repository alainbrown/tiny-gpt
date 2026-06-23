import torch
import torch.nn.functional as F
from torch import nn


class GPTModel(nn.Module):
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
                TransformerBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

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


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.ff2(F.gelu(self.ff1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.feed_forward = FeedForward(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x):
        attention = self.multi_head_attention(self.layer_norm1(x))
        x = x + self.dropout(attention)

        feed_forward = self.feed_forward(self.layer_norm2(x))
        return x + self.dropout(feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0, (
            "d_model must be divisible by n_heads"
        )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.head_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
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


Model = GPTModel


def convert_reference_state_dict(state_dict):
    """Convert reference Q/K/V and LayerNorm keys to the optimized layout."""
    converted = dict(state_dict)

    for key in list(converted):
        if key.endswith(".gamma"):
            converted[key.removesuffix(".gamma") + ".weight"] = converted.pop(
                key
            )
        elif key.endswith(".beta"):
            converted[key.removesuffix(".beta") + ".bias"] = converted.pop(
                key
            )

    attention_suffix = ".multi_head_attention.query.weight"
    query_weight_keys = [
        key for key in converted if key.endswith(attention_suffix)
    ]
    for query_weight_key in query_weight_keys:
        prefix = query_weight_key.removesuffix("query.weight")
        qkv_weight_key = prefix + "qkv.weight"
        qkv_bias_key = prefix + "qkv.bias"

        converted[qkv_weight_key] = torch.cat(
            [
                converted.pop(prefix + "query.weight"),
                converted.pop(prefix + "key.weight"),
                converted.pop(prefix + "value.weight"),
            ],
            dim=0,
        )
        converted[qkv_bias_key] = torch.cat(
            [
                converted.pop(prefix + "query.bias"),
                converted.pop(prefix + "key.bias"),
                converted.pop(prefix + "value.bias"),
            ],
            dim=0,
        )

    return converted

__all__ = [
    "convert_reference_state_dict",
    "FeedForward",
    "GPTModel",
    "Model",
    "MultiHeadAttention",
    "TransformerBlock",
]
