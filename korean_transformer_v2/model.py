"""V2 configuration and construction using the local Transformer implementation."""

from dataclasses import dataclass

from config import GPTConfig
from model import GPTLanguageModel
from model.attention import CausalSelfAttention


@dataclass
class V2Config(GPTConfig):
    vocab_size: int = 257
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    precision: str = "auto"
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.vocab_size < 257:
            raise ValueError("V2 vocabulary must contain the 256 bytes and EOS")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be between zero and max_steps")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between zero and one")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("invalid precision")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")


def build_model(config: V2Config) -> GPTLanguageModel:
    model = GPTLanguageModel(config)
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.use_sdpa = True
    return model
