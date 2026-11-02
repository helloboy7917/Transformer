import pytest
import torch

from korean_transformer_v2.model import V2Config, build_model


def test_v2_model_configuration_and_forward():
    config = V2Config(vocab_size=260, embedding_dim=16, num_heads=4, num_layers=1,
                      context_length=8, max_steps=10, warmup_steps=2)
    assert V2Config.from_dict(config.to_dict()) == config
    model = build_model(config)
    logits, loss = model(torch.randint(260, (2, 8)), torch.randint(260, (2, 8)))
    assert logits.shape == (2, 8, 260)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("kwargs", [{"vocab_size": 256}, {"precision": "invalid"},
                                  {"warmup_steps": 1001}, {"gradient_accumulation_steps": 0}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        V2Config(**kwargs)
