"""Generate text from a saved local GPT checkpoint."""

import argparse
from pathlib import Path

import torch

from config import GPTConfig, get_device
from data import CharTokenizer
from model import GPTLanguageModel
from utils.checkpoint import load_checkpoint, read_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "checkpoints" / "best.pt")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="always choose the highest-scoring token")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    # Load metadata first to build a model with matching dimensions and vocabulary.
    state = read_checkpoint(args.checkpoint, map_location="cpu")
    config = GPTConfig.from_dict(state["model_config"])
    tokenizer = CharTokenizer.from_dict(state["tokenizer"])
    model = GPTLanguageModel(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.unk_token_id]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    print(tokenizer.decode(generated[0]))


if __name__ == "__main__":
    main()
