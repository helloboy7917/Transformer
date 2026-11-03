from types import SimpleNamespace

import pytest
import torch

from korean_transformer_v2.inference import generate_text
from korean_transformer_v2.tokenizer import KoreanBPETokenizer


class StubModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(vocab_size=257, context_length=2)
        self.output = iter(output)
        self.context_sizes = []

    def forward(self, x):
        self.context_sizes.append(x.shape[1])
        logits = torch.full((1, x.shape[1], 257), -100.0)
        logits[0, -1, next(self.output)] = 100.0
        return logits


def test_generation_korean_eos_context_and_training_mode():
    tokenizer = KoreanBPETokenizer()
    model = StubModel([*"가".encode(), tokenizer.eos_token_id, 65])
    model.train()
    assert generate_text(model, tokenizer, "안녕", max_new_tokens=8, greedy=True) == "안녕가"
    assert model.training and model.context_sizes == [2, 2, 2, 2]


def test_generation_validates_and_supports_top_k_sampling():
    tokenizer = KoreanBPETokenizer()
    model = StubModel([65])
    assert generate_text(model, tokenizer, "문", max_new_tokens=1, top_k=1) == "문A"
    with pytest.raises(ValueError):
        generate_text(model, tokenizer, "", greedy=True)
