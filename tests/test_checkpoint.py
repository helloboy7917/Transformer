import torch

from config import GPTConfig
from data import CharTokenizer
from model import GPTLanguageModel
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_and_generation(tmp_path):
    tokenizer = CharTokenizer.from_text("hello world")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        context_length=8,
        feedforward_dim=32,
        dropout=0.0,
    )
    model = GPTLanguageModel(config).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        step=12,
        model_config=config.to_dict(),
        tokenizer=tokenizer.to_dict(),
        best_val_loss=1.23,
    )
    restored = GPTLanguageModel(config).eval()
    state = load_checkpoint(path, restored, map_location="cpu")
    assert state["step"] == 12
    torch.testing.assert_close(model(torch.tensor([[1, 2, 3]])), restored(torch.tensor([[1, 2, 3]])))
    generated = restored.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4, greedy=True)
    assert generated.shape == (1, 7)
