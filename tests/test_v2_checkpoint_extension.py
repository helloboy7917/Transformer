from dataclasses import replace
import torch
import pytest
from korean_transformer_v2.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from korean_transformer_v2.model import build_model
from korean_transformer_v2.scheduler import create_scheduler, learning_rate_multiplier
from korean_transformer_v2.tokenizer import KoreanBPETokenizer
from tests.test_v2_checkpoint import components


def test_extension_and_protected_history(tmp_path):
    model, optimizer, scheduler, runner = components()
    tokenizer = KoreanBPETokenizer()
    path = tmp_path / "model.pt"
    kwargs = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=runner.scaler,
                  tokenizer=tokenizer, protect=True)
    save_checkpoint(path, step=2, training_state={"tokens": 16}, **kwargs)
    old = path.read_bytes()
    save_checkpoint(path, step=3, **kwargs)
    assert next((tmp_path / "model_history").iterdir()).read_bytes() == old
    state = read_checkpoint(path)
    extended = build_model(replace(model.config, max_steps=10))
    opt = torch.optim.AdamW(extended.parameters(), lr=model.config.learning_rate)
    sched = create_scheduler(opt, total_steps=10, warmup_steps=2)
    assert restore_checkpoint(state, model=extended, optimizer=opt, scheduler=sched,
                              scaler=runner.scaler, tokenizer=tokenizer) == 3
    assert opt.param_groups[0]["lr"] == pytest.approx(model.config.learning_rate * learning_rate_multiplier(
        3, total_steps=10, warmup_steps=2, min_lr_ratio=0.1))
