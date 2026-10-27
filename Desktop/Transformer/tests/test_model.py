import torch

from config import GPTConfig
from model import GPTLanguageModel, PositionalEmbedding, TokenEmbedding, TransformerBlock


def tiny_config(**overrides):
    values = dict(
        vocab_size=19,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        context_length=16,
        feedforward_dim=64,
        dropout=0.0,
    )
    values.update(overrides)
    return GPTConfig(**values)


def test_embedding_and_block_shapes():
    ids = torch.randint(0, 19, (2, 7))
    assert TokenEmbedding(19, 32)(ids).shape == (2, 7, 32)
    assert PositionalEmbedding(16, 32)(7).shape == (7, 32)
    block = TransformerBlock(32, 4, 64, 0.0)
    assert block(torch.randn(2, 7, 32)).shape == (2, 7, 32)


def test_logits_loss_and_gradient_shapes():
    model = GPTLanguageModel(tiny_config())
    x = torch.randint(0, 19, (2, 8))
    y = torch.randint(0, 19, (2, 8))
    logits = model(x)
    logits_with_targets, loss = model(x, y)
    assert logits.shape == (2, 8, 19)
    assert logits_with_targets.shape == logits.shape
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.grad is not None


def test_attention_heads_must_divide_embedding_dimension():
    try:
        GPTConfig(vocab_size=10, embedding_dim=30, num_heads=4)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid head count was accepted")
