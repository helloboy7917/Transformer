"""Checkpoint I/O with atomic writes to avoid partially written model files."""

from pathlib import Path
import os

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    *,
    step: int,
    model_config: dict,
    tokenizer: dict,
    best_val_loss: float | None = None,
    metadata: dict | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "model_config": model_config,
        "tokenizer": tokenizer,
        "best_val_loss": best_val_loss,
        "metadata": metadata or {},
    }
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions before weights_only.
        return torch.load(path, map_location=map_location)


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> dict:
    checkpoint = read_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
