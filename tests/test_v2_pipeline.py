import pytest
import torch

from korean_transformer_v2.checkpoint import read_checkpoint
from korean_transformer_v2.model import V2Config
from korean_transformer_v2.pipeline import train_documents


def test_training_resume_matches_uninterrupted_run(tmp_path):
    docs = [f"한국어 학습 문장 번호 {i} 테스트" for i in range(30)]
    config = V2Config(embedding_dim=8, num_heads=2, num_layers=1, context_length=4,
                      batch_size=2, max_steps=3, warmup_steps=1, eval_interval=2,
                      eval_batches=1, checkpoint_interval=1, dropout=0.1)
    full, partial = tmp_path / "full.pt", tmp_path / "partial.pt"
    train_documents(docs, config=config, output_path=full, vocab_size=265)
    assert train_documents(docs, config=config, output_path=partial, vocab_size=265, stop_after=1)["step"] == 1
    assert train_documents(docs, config=config, output_path=partial, resume_path=partial)["step"] == 3
    a, b = read_checkpoint(full), read_checkpoint(partial)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["scheduler"] == b["scheduler"]
    with pytest.raises(ValueError, match="fingerprint"):
        train_documents(docs + ["새 문서"], config=config, output_path=partial, resume_path=partial)
    with pytest.raises(FileExistsError):
        train_documents(docs, config=config, output_path=full)
