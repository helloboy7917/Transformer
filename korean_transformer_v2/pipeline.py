"""Connect V2 data, training, evaluation and checkpoint recovery."""

from dataclasses import replace
import json
from pathlib import Path
import random
import time
import uuid

import torch

from data.dataset import get_batch
from .checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from .corpus import DiskCorpus, load_prepared, save_prepared
from .model import build_model
from .optimization import TrainingStep
from .scheduler import create_scheduler
from .tokenizer import FastBPETokenizer, KoreanBPETokenizer


def train_documents(documents, *, config, output_path, vocab_size=512, device="cpu",
                    resume_path=None, stop_after=None, cache_dir=None,
                    tokenizer_backend="python", tokenizer_path=None,
                    tokenizer_max_documents=20000, tokenizer_max_characters=20000000,
                    log_path=None, prepared_cache=None):
    """Prepare on disk and train to an absolute step using CPU-mapped tokens."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive for training")
    if tokenizer_backend not in {"python", "fast"}:
        raise ValueError("invalid tokenizer backend")
    if tokenizer_max_documents <= 0 or tokenizer_max_characters <= 0:
        raise ValueError("tokenizer sample limits must be positive")
    if stop_after is not None and not 0 <= stop_after <= config.max_steps:
        raise ValueError("stop_after must be between zero and max_steps")
    if Path(output_path).exists() and (resume_path is None or Path(output_path).resolve() != Path(resume_path).resolve()):
        raise FileExistsError("output checkpoint already exists; resume that checkpoint or choose a new output")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    state = read_checkpoint(resume_path) if resume_path is not None else None
    root = Path(cache_dir) if cache_dir else Path(output_path).parent / (Path(output_path).stem + ".cache")
    if prepared_cache:
        fingerprint, tokenizer, train_tokens, val_tokens = load_prepared(prepared_cache, config)
        cache_path = Path(prepared_cache)
        if state is not None and state["tokenizer"] != tokenizer.to_dict():
            raise ValueError("prepared corpus tokenizer differs from checkpoint")
        if tokenizer_path and KoreanBPETokenizer.load(tokenizer_path).to_dict() != tokenizer.to_dict():
            raise ValueError("prepared corpus tokenizer differs from tokenizer file")
    else:
        corpus = DiskCorpus(documents, root / uuid.uuid4().hex,
                            validation_fraction=config.validation_fraction, seed=config.seed)
        try:
            fingerprint = corpus.fingerprint
            if state is not None:
                tokenizer = KoreanBPETokenizer.from_dict(state["tokenizer"])
                if tokenizer_path and KoreanBPETokenizer.load(tokenizer_path).to_dict() != tokenizer.to_dict():
                    raise ValueError("checkpoint tokenizer differs from tokenizer file")
            elif tokenizer_path:
                tokenizer = KoreanBPETokenizer.load(tokenizer_path)
            else:
                sample_path = corpus.directory / "tokenizer-sample.txt"
                corpus.write_tokenizer_sample(sample_path, max_documents=tokenizer_max_documents,
                                              max_characters=tokenizer_max_characters)
                if tokenizer_backend == "fast":
                    tokenizer = FastBPETokenizer.train_file(sample_path, vocab_size=vocab_size)
                else:
                    with sample_path.open(encoding="utf-8") as sample:
                        tokenizer = KoreanBPETokenizer.train(
                            (line.removesuffix("\n") for line in sample), vocab_size=vocab_size)
            train_tokens = corpus.encode(tokenizer, "train")
            val_tokens = corpus.encode(tokenizer, "validation")
            save_prepared(corpus.directory, fingerprint=fingerprint, tokenizer=tokenizer, config=config,
                          train_tokens=train_tokens, val_tokens=val_tokens)
        finally:
            corpus.close()
        cache_path = corpus.directory
    config = replace(config, vocab_size=tokenizer.vocab_size)
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
    tokens_seen = (state or {}).get("training_state", {}).get("tokens_seen", 0)
    log_path = Path(log_path) if log_path else Path(output_path).with_suffix(".metrics.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        save_checkpoint(output_path, model=model, optimizer=optimizer, scheduler=scheduler,
                        scaler=runner.scaler, tokenizer=tokenizer, step=step, data_fingerprint=fingerprint,
                        protect=True, training_state={"tokens_seen": tokens_seen, "precision": runner.precision})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target = config.max_steps if stop_after is None else stop_after
    if target < step:
        raise ValueError("stop_after precedes the checkpoint step")
    last_loss = None
    consecutive_overflows = 0
    while step < target:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        batches = [get_batch(train_tokens, config.batch_size, config.context_length)
                   for _ in range(config.gradient_accumulation_steps)]
        last_loss, updated = runner.run(batches)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if not updated:
            consecutive_overflows += 1
            if consecutive_overflows >= 10:
                raise RuntimeError("ten consecutive mixed-precision optimizer overflows")
            continue
        consecutive_overflows = 0
        step += 1
        scheduler.step()
        token_count = config.batch_size * config.context_length * config.gradient_accumulation_steps
        tokens_seen += token_count
        validation_loss = None
        if step % config.eval_interval == 0 or step == config.max_steps:
            model.eval()
            with torch.inference_mode(), runner.autocast():
                losses = []
                for _ in range(config.eval_batches):
                    x, y = get_batch(val_tokens, config.batch_size, config.context_length)
                    losses.append(model(x.to(device), y.to(device))[1].float().item())
            validation_loss = sum(losses) / len(losses)
        metrics = {"step": step, "train_loss": last_loss, "validation_loss": validation_loss,
                   "tokens_per_second": token_count / elapsed, "tokens_seen": tokens_seen,
                   "lr": optimizer.param_groups[0]["lr"], "precision": runner.precision,
                   "gpu_allocated_mb": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
                   "gpu_reserved_mb": torch.cuda.memory_reserved(device) / 2**20 if device.type == "cuda" else 0,
                   "gpu_peak_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(metrics) + "\n")
        if validation_loss is not None:
            print(f"step={step}/{config.max_steps} train_loss={last_loss:.4f} validation_loss={validation_loss:.4f} tokens/sec={metrics['tokens_per_second']:.0f} gpu_peak_mb={metrics['gpu_peak_mb']:.0f}")
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save()
    save()
    return {"step": step, "loss": last_loss, "config": config.to_dict(), "checkpoint": str(output_path),
            "metrics": str(log_path), "cache": str(cache_path)}
