"""Train the from-scratch character GPT on a UTF-8 text corpus."""

import argparse
import csv
from pathlib import Path
import random

import torch

from config import GPTConfig, get_device
from data import CharTokenizer, get_batch, split_token_ids
from model import GPTLanguageModel
from model.gpt import count_parameters
from utils.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "sample.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def estimate_loss(model, train_tokens, validation_tokens, config):
    model.eval()
    result = {}
    for name, tokens in (("train", train_tokens), ("val", validation_tokens)):
        losses = []
        for _ in range(config.eval_batches):
            x, y = get_batch(tokens, config.batch_size, config.context_length)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="UTF-8 .txt training corpus")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, or a specific torch device")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.context_length is not None and args.context_length <= 0:
        raise ValueError("context_length must be positive")
    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if args.checkpoint_interval is not None and args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    device = get_device(args.device)
    resume_state = read_checkpoint(args.resume, map_location="cpu") if args.resume else None

    text = args.data.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"training file is empty: {args.data}")
    tokenizer = (
        CharTokenizer.from_dict(resume_state["tokenizer"])
        if resume_state
        else CharTokenizer.from_text(text)
    )
    validation_fraction = (
        resume_state.get("model_config", {}).get("validation_fraction", GPTConfig().validation_fraction)
        if resume_state
        else GPTConfig().validation_fraction
    )
    train_ids, validation_ids = split_token_ids(tokenizer.encode(text), validation_fraction)

    if resume_state:
        config = GPTConfig.from_dict(resume_state["model_config"])
    else:
        config = GPTConfig(vocab_size=tokenizer.vocab_size)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.context_length is not None:
        if resume_state and args.context_length != config.context_length:
            raise ValueError("context length cannot change when resuming a checkpoint")
        config.context_length = args.context_length
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    config.vocab_size = tokenizer.vocab_size

    # A short custom corpus can still train: shrink the window to fit both splits.
    usable_context = min(config.context_length, len(train_ids) - 1, len(validation_ids) - 1)
    if usable_context < 1:
        raise ValueError("training and validation splits must each contain at least two characters")
    if resume_state and usable_context < config.context_length:
        raise ValueError("the resumed corpus is too short for the checkpoint context length")
    if usable_context != config.context_length:
        print(f"Corpus is short; context length adjusted from {config.context_length} to {usable_context}.")
        config.context_length = usable_context

    set_seed(config.seed)
    train_tokens = train_ids.to(device)
    validation_tokens = validation_ids.to(device)
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_step = 0
    best_val_loss = float("inf")
    if resume_state:
        checkpoint = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_step = int(checkpoint.get("step", 0))
        saved_best_loss = checkpoint.get("best_val_loss")
        best_val_loss = saved_best_loss if saved_best_loss is not None else float("inf")

    print(f"Device: {device} | vocabulary: {config.vocab_size} | parameters: {count_parameters(model):,}")
    print(
        f"Training characters: {len(train_tokens):,} | validation characters: {len(validation_tokens):,} | "
        f"context: {config.context_length}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            csv.writer(log_file).writerow(("step", "train_loss", "validation_loss", "learning_rate"))
    model.train()
    for step in range(start_step + 1, config.max_steps + 1):
        x, y = get_batch(train_tokens, config.batch_size, config.context_length)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        should_evaluate = step == 1 or step % config.eval_interval == 0 or step == config.max_steps
        metrics = None
        if should_evaluate:
            metrics = estimate_loss(model, train_tokens, validation_tokens, config)
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:>6} | train loss {metrics['train']:.4f} | "
                f"val loss {metrics['val']:.4f} | lr {learning_rate:.6g}"
            )
            with log_path.open("a", newline="", encoding="utf-8") as log_file:
                csv.writer(log_file).writerow((step, metrics["train"], metrics["val"], learning_rate))

        checkpoint_args = dict(
            step=step,
            model_config=config.to_dict(),
            tokenizer=tokenizer.to_dict(),
            best_val_loss=best_val_loss,
        )
        if metrics and metrics["val"] < best_val_loss:
            best_val_loss = metrics["val"]
            checkpoint_args["best_val_loss"] = best_val_loss
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, **checkpoint_args
            )
        if config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0:
            save_checkpoint(
                args.output_dir / f"step_{step:07d}.pt", model, optimizer, **checkpoint_args
            )
        if step % args.log_interval == 0 or step == config.max_steps:
            save_checkpoint(args.output_dir / "last.pt", model, optimizer, **checkpoint_args)

    # Keep a final checkpoint even when max_steps is zero or resume already reached it.
    if start_step >= config.max_steps:
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            step=start_step,
            model_config=config.to_dict(),
            tokenizer=tokenizer.to_dict(),
            best_val_loss=best_val_loss,
        )
    print(f"Saved checkpoints in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
