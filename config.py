"""Central configuration for the from-scratch character GPT."""

from dataclasses import asdict, dataclass, fields


@dataclass
class GPTConfig:
    # Model dimensions
    vocab_size: int = 0  # Filled from the tokenizer before model construction.
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    context_length: int = 128
    feedforward_dim: int | None = None  # Defaults to 4 * embedding_dim.
    dropout: float = 0.1

    # Training settings
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 500
    grad_clip: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.feedforward_dim is None:
            self.feedforward_dim = 4 * self.embedding_dim
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.num_heads <= 0 or self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by a positive num_heads")
        if self.num_layers <= 0 or self.context_length <= 0:
            raise ValueError("num_layers and context_length must be positive")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive")
        if self.max_steps < 0 or self.eval_interval <= 0 or self.eval_batches <= 0:
            raise ValueError("max_steps must be non-negative; evaluation intervals must be positive")
        if self.checkpoint_interval < 0 or self.grad_clip <= 0:
            raise ValueError("checkpoint_interval must be non-negative and grad_clip must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "GPTConfig":
        """Load known fields only, allowing checkpoints to carry extra metadata."""
        accepted = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in accepted})


def get_device(requested: str | None = None):
    """Use an explicitly requested device, or CUDA when it is available."""
    import torch

    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
