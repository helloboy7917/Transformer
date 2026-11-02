import random

import pytest
import torch

from korean_transformer_v2.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from korean_transformer_v2.model import V2Config, build_model
from korean_transformer_v2.optimization import TrainingStep
from korean_transformer_v2.scheduler import create_scheduler
from korean_transformer_v2.tokenizer import KoreanBPETokenizer


def components():
    config = V2Config(embedding_dim=8, num_heads=2, num_layers=1, context_length=4, max_steps=5, warmup_steps=2)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = create_scheduler(optimizer, total_steps=5, warmup_steps=2)
    runner = TrainingStep(model, optimizer, device="cpu")
    return model, optimizer, scheduler, runner


def test_checkpoint_restores_training_and_rng(tmp_path):
    tokenizer = KoreanBPETokenizer()
    model, optimizer, scheduler, runner = components()
    x = torch.randint(257, (2, 4))
    runner.run([(x, x)])
    scheduler.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=runner.scaler, tokenizer=tokenizer, step=1)
    expected_torch, expected_python = torch.rand(3), random.random()
    restored, opt2, sched2, runner2 = components()
    state = read_checkpoint(path)
    assert restore_checkpoint(state, model=restored, optimizer=opt2, scheduler=sched2, scaler=runner2.scaler, tokenizer=tokenizer) == 1
    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert random.random() == expected_python
    assert sched2.state_dict() == scheduler.state_dict()
    assert opt2.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for p1, p2 in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p1, p2)
    with pytest.raises(ValueError, match="tokenizer"):
        restore_checkpoint(state, model=restored, optimizer=opt2, scheduler=sched2, scaler=runner2.scaler, tokenizer=KoreanBPETokenizer([(1, 2)]))
