"""Shared training loop used by the standalone and multi-model CLIs."""

import csv
from pathlib import Path
import random

import torch

from data.dataset import get_batch
from utils.checkpoint import save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def estimate_loss(model, train_tokens, validation_tokens, config) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        result = {}
        for name, tokens in (("train", train_tokens), ("validation", validation_tokens)):
            losses = []
            for _ in range(config.eval_batches):
                x, y = get_batch(tokens, config.batch_size, config.context_length)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
        return result
    finally:
        model.train(was_training)


def train_loop(
    model,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    config,
    tokenizer_state: dict,
    output_dir: Path,
    *,
    optimizer,
    device: torch.device,
    start_step: int = 0,
    best_val_loss: float = float("inf"),
    model_name: str | None = None,
    checkpoint_metadata: dict | None = None,
    last_save_interval: int | None = None,
) -> dict:
    """Run optimizer steps and persist compatible resumable checkpoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            csv.writer(log_file).writerow(("step", "train_loss", "validation_loss", "learning_rate"))

    if last_save_interval is None:
        last_save_interval = config.checkpoint_interval or config.eval_interval
    if last_save_interval <= 0:
        last_save_interval = config.eval_interval

    if model_name:
        print(f"Model: {model_name}")
    print(f"Device: {device}")
    total_steps = config.max_steps
    model.train()
    last_validation_loss = None

    for step in range(start_step + 1, total_steps + 1):
        x, y = get_batch(train_tokens, config.batch_size, config.context_length)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        should_evaluate = step == start_step + 1 or step % config.eval_interval == 0 or step == total_steps
        metrics = None
        if should_evaluate:
            metrics = estimate_loss(model, train_tokens, validation_tokens, config)
            last_validation_loss = metrics["validation"]
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"Step: {step}/{total_steps}\n"
                f"Train Loss: {metrics['train']:.4f}\n"
                f"Validation Loss: {metrics['validation']:.4f}\n"
                f"Learning Rate: {learning_rate:.6g}"
            )
            with log_path.open("a", newline="", encoding="utf-8") as log_file:
                csv.writer(log_file).writerow(
                    (step, metrics["train"], metrics["validation"], learning_rate)
                )

        metadata = dict(checkpoint_metadata or {})
        if metrics:
            metadata["last_validation_loss"] = metrics["validation"]
        checkpoint_args = dict(
            step=step,
            model_config=config.to_dict(),
            tokenizer=tokenizer_state,
            best_val_loss=best_val_loss,
            metadata=metadata,
        )
        if metrics and metrics["validation"] < best_val_loss:
            best_val_loss = metrics["validation"]
            checkpoint_args["best_val_loss"] = best_val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, **checkpoint_args)
        if config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0:
            save_checkpoint(
                output_dir / f"step_{step:07d}.pt", model, optimizer, **checkpoint_args
            )
        if step % last_save_interval == 0 or step == total_steps:
            save_checkpoint(output_dir / "last.pt", model, optimizer, **checkpoint_args)

    if start_step >= total_steps:
        metadata = dict(checkpoint_metadata or {})
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            step=start_step,
            model_config=config.to_dict(),
            tokenizer=tokenizer_state,
            best_val_loss=best_val_loss,
            metadata=metadata,
        )
    return {
        "step": max(start_step, total_steps),
        "best_val_loss": best_val_loss,
        "last_validation_loss": last_validation_loss,
    }
