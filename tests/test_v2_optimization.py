import copy

import pytest
import torch

from korean_transformer_v2.model import V2Config, build_model
from korean_transformer_v2.optimization import TrainingStep


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(2)
    config = V2Config(embedding_dim=8, num_heads=2, num_layers=1, context_length=4, dropout=0)
    first = build_model(config)
    second = copy.deepcopy(first)
    x = torch.randint(257, (4, 4))
    y = torch.randint(257, (4, 4))
    optim1 = torch.optim.SGD(first.parameters(), lr=0.01)
    optim2 = torch.optim.SGD(second.parameters(), lr=0.01)
    runner1 = TrainingStep(first, optim1, device="cpu", grad_clip=100)
    runner2 = TrainingStep(second, optim2, device="cpu", grad_clip=100)
    loss1, stepped = runner1.run([(x[:2], y[:2]), (x[2:], y[2:])])
    loss2, _ = runner2.run([(x, y)])
    assert stepped and loss1 == pytest.approx(loss2, abs=1e-6)
    for p1, p2 in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p1, p2)
    with pytest.raises(ValueError):
        runner1.run([])


def test_cpu_rejects_cuda_precision():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        TrainingStep(model, optimizer, device="cpu", precision="fp16")
