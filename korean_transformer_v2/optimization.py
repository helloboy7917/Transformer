"""Mixed-precision optimizer steps with gradient accumulation and clipping."""

from contextlib import nullcontext

import torch


class TrainingStep:
    def __init__(self, model, optimizer, *, device, precision="auto", grad_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("invalid precision")
        if grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.device.type != "cuda" and precision in {"fp16", "bf16"}:
            raise ValueError("mixed precision currently requires CUDA")
        if precision == "auto":
            precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if self.device.type == "cuda" else "fp32"
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bfloat16")
        self.precision = precision
        self.grad_clip = grad_clip
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        else:  # PyTorch 2.0/2.1 compatibility.
            self.scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")

    def run(self, batches):
        """Perform one optimizer step on an iterable of equally sized microbatches."""
        batches = list(batches)
        if not batches:
            raise ValueError("at least one microbatch is required")
        sizes = [x.shape[0] for x, _ in batches]
        if not all(size > 0 and size == sizes[0] for size in sizes):
            raise ValueError("microbatches must have equal positive batch sizes")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for x, y in batches:
            x, y = x.to(self.device), y.to(self.device)
            context = (nullcontext() if self.precision == "fp32" else
                       torch.autocast(device_type="cuda", dtype=torch.float16 if self.precision == "fp16" else torch.bfloat16))
            with context:
                _, loss = self.model(x, y)
            total_loss += loss.detach().float().item()
            self.scaler.scale(loss / len(batches)).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return total_loss / len(batches), self.scaler.get_scale() >= previous_scale
