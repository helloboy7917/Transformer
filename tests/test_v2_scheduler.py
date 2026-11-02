import pytest
import torch

from korean_transformer_v2.scheduler import create_scheduler, learning_rate_multiplier


def test_warmup_cosine_decay_and_scheduler_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1)
    scheduler = create_scheduler(optimizer, total_steps=6, warmup_steps=2, min_lr_ratio=0.1)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0:2] == [0.5, 1.0]
    assert rates[2] > rates[3] > rates[4] > rates[5]
    assert rates[-1] == pytest.approx(0.1)
    assert scheduler.state_dict()["last_epoch"] == 6


def test_scheduler_invalid_arguments_and_zero_warmup():
    assert learning_rate_multiplier(20, total_steps=4, warmup_steps=0, min_lr_ratio=0.2) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        learning_rate_multiplier(0, total_steps=2, warmup_steps=3, min_lr_ratio=0.1)
