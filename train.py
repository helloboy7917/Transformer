"""Train the from-scratch character GPT on a UTF-8 text corpus."""

import argparse
from pathlib import Path

import torch

from config import GPTConfig, get_device
from data import CharTokenizer, split_token_ids
from model import GPTLanguageModel
from model.gpt import count_parameters
from services.trainer import set_seed, train_loop
from utils.checkpoint import load_checkpoint, read_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "sample.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "checkpoints"


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
        f"context: {config.context_length} | parameters: {count_parameters(model):,}"
    )
    train_loop(
        model,
        train_tokens,
        validation_tokens,
        config,
        tokenizer.to_dict(),
        args.output_dir,
        optimizer=optimizer,
        device=device,
        start_step=start_step,
        best_val_loss=best_val_loss,
        last_save_interval=args.log_interval,
    )
    print(f"Saved checkpoints in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
