"""Autoregressive Korean text generation with end-of-document stopping."""

import torch


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, *, max_new_tokens: int = 100,
                  temperature: float = 1.0, top_k: int | None = None,
                  greedy: bool = False) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0 or temperature <= 0 or (top_k is not None and top_k <= 0):
        raise ValueError("invalid generation settings")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = torch.tensor([tokens[-model.config.context_length:]], dtype=torch.long, device=device)
            logits = model(context)[0, -1, :]
            if greedy:
                token = int(logits.argmax().item())
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.numel())
                    indices = torch.topk(logits, k).indices
                    filtered = torch.full_like(logits, float("-inf"))
                    logits = filtered.scatter(0, indices, logits[indices])
                token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            if token == tokenizer.eos_token_id:
                break
            tokens.append(token)
        return tokenizer.decode(tokens)
    finally:
        model.train(was_training)
