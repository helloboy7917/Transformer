"""Token and learned position embeddings."""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, time) -> vectors: (batch, time, embedding_dim)
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, sequence_length: int, device: torch.device | None = None) -> torch.Tensor:
        if sequence_length > self.embedding.num_embeddings:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length "
                f"{self.embedding.num_embeddings}"
            )
        positions = torch.arange(sequence_length, device=device)
        # positions: (time,) -> vectors: (time, embedding_dim)
        return self.embedding(positions)


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embedding_dim)
        self.position = PositionalEmbedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape
        token_vectors = self.token(token_ids)  # (batch, time, embedding_dim)
        position_vectors = self.position(sequence_length, token_ids.device)  # (time, embedding_dim)
        return self.dropout(token_vectors + position_vectors.unsqueeze(0))
