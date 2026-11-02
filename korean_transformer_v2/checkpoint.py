"""Atomic, resumable V2 checkpoints including optimizer and random state."""

import os
from pathlib import Path
import random
import tempfile

import torch


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, tokenizer, step: int,
                    data_fingerprint: str | None = None):
    if step < 0:
        raise ValueError("step must be nonnegative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "korean_transformer_v2_1", "step": step, "data_fingerprint": data_fingerprint,
        "config": model.config.to_dict(), "tokenizer": tokenizer.to_dict(),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(state, handle)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_checkpoint(path, *, map_location="cpu"):
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        state = torch.load(path, map_location=map_location)
    if state.get("format") != "korean_transformer_v2_1":
        raise ValueError("not a supported V2 checkpoint")
    return state


def restore_checkpoint(state, *, model, optimizer, scheduler, scaler, tokenizer,
                       data_fingerprint: str | None = None):
    if state.get("format") != "korean_transformer_v2_1":
        raise ValueError("not a supported V2 checkpoint")
    if state["config"] != model.config.to_dict() or state["tokenizer"] != tokenizer.to_dict():
        raise ValueError("checkpoint config or tokenizer does not match")
    if state.get("data_fingerprint") != data_fingerprint:
        raise ValueError("checkpoint data fingerprint does not match")
    if state["cuda_rng"] and torch.cuda.is_available() and len(state["cuda_rng"]) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA device count does not match")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    random.setstate(state["python_rng"])
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    return state["step"]
