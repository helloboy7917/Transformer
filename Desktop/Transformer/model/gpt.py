"""Decoder-only GPT language model assembled from the local components."""

import torch
from torch import nn
from torch.nn import functional as F

from config import GPTConfig
from .embedding import TokenPositionEmbedding
from .transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("config.vocab_size must be set from the tokenizer")
        self.config = config
        self.embedding = TokenPositionEmbedding(
            config.vocab_size, config.context_length, config.embedding_dim, config.dropout
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.feedforward_dim,
                config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """Return logits (B,T,V), and optionally mean next-token cross entropy."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, time)")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence exceeds the model context length")
        x = self.embedding(token_ids)  # (batch, time, embedding_dim)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))  # (batch, time, vocab_size)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Autoregressively append tokens, cropping prompts to the context window."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = token_ids[:, -self.config.context_length :]
                logits = self(context)[:, -1, :]  # (batch, vocab_size)
                if greedy:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(top_k, logits.shape[-1])
                        selected = torch.topk(logits, k, dim=-1).indices
                        filtered = torch.full_like(logits, float("-inf"))
                        logits = filtered.scatter(1, selected, logits.gather(1, selected))
                    probabilities = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                token_ids = torch.cat((token_ids, next_token), dim=1)
            return token_ids
        finally:
            self.train(was_training)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
