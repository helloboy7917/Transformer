import torch

from config import GPTConfig
from data import CharTokenizer, get_batch
from model import GPTLanguageModel


def test_tiny_model_can_overfit_a_small_repeating_corpus():
    torch.manual_seed(7)
    text = ("abcabcabcabcabcabc" * 10)
    tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        context_length=12,
        feedforward_dim=32,
        dropout=0.0,
        batch_size=8,
        learning_rate=0.01,
    )
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    x, y = get_batch(tokens, 8, config.context_length)
    with torch.no_grad():
        initial_loss = model(x, y)[1].item()
    for _ in range(45):
        x, y = get_batch(tokens, 8, config.context_length)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = model(x, y)[1].item()
    assert final_loss < initial_loss * 0.6
