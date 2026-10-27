"""Pre-LayerNorm Transformer block with residual connections."""

from torch import nn

from .attention import CausalSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, feedforward_dim, dropout)

    def forward(self, x):
        # x: (batch, time, embedding_dim)
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x
