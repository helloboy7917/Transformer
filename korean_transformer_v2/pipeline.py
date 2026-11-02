"""Connect V2 data, training, evaluation and checkpoint recovery."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import torch

from data.dataset import get_batch
from .checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from .model import build_model
from .optimization import TrainingStep
from .preprocessing import encode_documents, prepare_documents
from .scheduler import create_scheduler
from .tokenizer import KoreanBPETokenizer


def train_documents(documents, *, config, output_path, vocab_size=512, device="cpu",
                    resume_path=None, stop_after=None):
    """Train up to an absolute step; documents should be bounded by the caller."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive for training")
    if stop_after is not None and not 0 <= stop_after <= config.max_steps:
        raise ValueError("stop_after must be between zero and max_steps")
    if Path(output_path).exists() and resume_path is None:
        raise FileExistsError("output checkpoint already exists; specify resume_path")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_docs, val_docs = prepare_documents(documents, validation_fraction=config.validation_fraction, seed=config.seed)
    fingerprint = hashlib.sha256(json.dumps([train_docs, val_docs], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    state = read_checkpoint(resume_path) if resume_path is not None else None
    tokenizer = (KoreanBPETokenizer.from_dict(state["tokenizer"]) if state is not None
                 else KoreanBPETokenizer.train(train_docs, vocab_size=vocab_size))
    config = replace(config, vocab_size=tokenizer.vocab_size)
    train_tokens = torch.tensor(encode_documents(train_docs, tokenizer), dtype=torch.long, device=device)
    val_tokens = torch.tensor(encode_documents(val_docs, tokenizer), dtype=torch.long, device=device)
    if min(len(train_tokens), len(val_tokens)) <= config.context_length:
        raise ValueError("each token split must be longer than context_length")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, total_steps=config.max_steps, warmup_steps=config.warmup_steps, min_lr_ratio=config.min_lr_ratio)
    runner = TrainingStep(model, optimizer, device=device, precision=config.precision, grad_clip=config.grad_clip)
    step = 0
    if state is not None:
        step = restore_checkpoint(state, model=model, optimizer=optimizer, scheduler=scheduler,
                                  scaler=runner.scaler, tokenizer=tokenizer, data_fingerprint=fingerprint)
    target = config.max_steps if stop_after is None else stop_after
    if target < step:
        raise ValueError("stop_after precedes the checkpoint step")
    last_loss = None
    consecutive_overflows = 0
    while step < target:
        batches = [get_batch(train_tokens, config.batch_size, config.context_length)
                   for _ in range(config.gradient_accumulation_steps)]
        last_loss, updated = runner.run(batches)
        if not updated:
            consecutive_overflows += 1
            if consecutive_overflows >= 10:
                raise RuntimeError("ten consecutive mixed-precision optimizer overflows")
            continue
        consecutive_overflows = 0
        step += 1
        scheduler.step()
        if step % config.eval_interval == 0 or step == config.max_steps:
            model.eval()
            with torch.inference_mode():
                losses = [model(*get_batch(val_tokens, config.batch_size, config.context_length))[1].item()
                          for _ in range(config.eval_batches)]
            print(f"step={step}/{config.max_steps} train_loss={last_loss:.4f} validation_loss={sum(losses)/len(losses):.4f} lr={optimizer.param_groups[0]['lr']:.8g}")
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save_checkpoint(output_path, model=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=runner.scaler, tokenizer=tokenizer, step=step, data_fingerprint=fingerprint)
    save_checkpoint(output_path, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=runner.scaler, tokenizer=tokenizer, step=step, data_fingerprint=fingerprint)
    return {"step": step, "loss": last_loss, "config": config.to_dict(), "checkpoint": str(output_path)}
