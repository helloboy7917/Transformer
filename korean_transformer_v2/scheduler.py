"""Linear warmup followed by cosine learning-rate decay."""

import math

import torch


def learning_rate_multiplier(step: int, *, total_steps: int, warmup_steps: int,
                             min_lr_ratio: float) -> float:
    if total_steps <= 0 or not 0 <= warmup_steps <= total_steps:
        raise ValueError("invalid total_steps or warmup_steps")
    if step < 0 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("invalid step or minimum learning-rate ratio")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps == 0:
        return 1.0
    progress = min(1.0, (step - warmup_steps + 1) / decay_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def create_scheduler(optimizer, *, total_steps: int, warmup_steps: int,
                     min_lr_ratio: float = 0.1):
    learning_rate_multiplier(0, total_steps=total_steps, warmup_steps=warmup_steps,
                             min_lr_ratio=min_lr_ratio)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_multiplier(
            step, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=min_lr_ratio
        )
    )
