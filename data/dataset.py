"""Next-character windows and random training batches."""

import torch
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    def __init__(self, token_ids, context_length: int) -> None:
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        if context_length <= 0:
            raise ValueError("context_length must be positive")

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        x = self.token_ids[index : index + self.context_length]
        y = self.token_ids[index + 1 : index + self.context_length + 1]
        return x, y


def split_token_ids(token_ids, validation_fraction: float = 0.1):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    tokens = torch.as_tensor(token_ids, dtype=torch.long)
    if len(tokens) < 4:
        raise ValueError("at least four tokens are required for a train/validation split")
    split_at = min(max(int(len(tokens) * (1.0 - validation_fraction)), 2), len(tokens) - 2)
    return tokens[:split_at], tokens[split_at:]


def get_batch(token_ids: torch.Tensor, batch_size: int, context_length: int):
    """Sample x and one-position-shifted y windows on the token tensor's device."""
    if len(token_ids) <= context_length:
        raise ValueError("token split must be longer than context_length")
    starts = torch.randint(len(token_ids) - context_length, (batch_size,), device=token_ids.device)
    offsets = torch.arange(context_length, device=token_ids.device)
    positions = starts[:, None] + offsets[None, :]
    x = token_ids[positions]
    y = token_ids[positions + 1]
    return x, y
