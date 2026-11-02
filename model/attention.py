"""Manual scaled dot-product attention and multi-head causal self-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
):
    """Compute softmax(QK^T / sqrt(d_k) + mask)V for (B, H, T, D) tensors."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (batch, heads, time, head_dim)")
    if query.shape[:2] != key.shape[:2] or key.shape != value.shape:
        raise ValueError("query/key/value batch, head, and key-time dimensions must be compatible")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")

    # (batch, heads, query_time, key_time)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if causal:
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("causal self-attention expects equal query and key lengths")
        length = query.shape[-2]
        allowed = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=training)
    output = torch.matmul(weights, value)  # (batch, heads, query_time, head_dim)
    return (output, weights) if return_weights else output


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = dropout
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        # One projection, then split channels into heads: (B, H, T, head_dim).
        qkv = self.qkv_projection(x)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            causal=True,
            dropout_p=self.attention_dropout,
            training=self.training,
        )
        # Join head outputs: (B, H, T, head_dim) -> (B, T, embedding_dim).
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.residual_dropout(self.output_projection(attended))
