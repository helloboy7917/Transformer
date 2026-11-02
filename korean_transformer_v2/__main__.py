"""Train and generate text with Korean Transformer V2."""

import argparse

from config import get_device
from .checkpoint import read_checkpoint
from .data_loader import stream_huggingface, stream_local
from .inference import generate_text
from .model import V2Config, build_model
from .pipeline import train_documents
from .tokenizer import KoreanBPETokenizer


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train or resume a model")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="UTF-8 text file, one document per line")
    source.add_argument("--dataset", help="Hugging Face dataset repository ID")
    train.add_argument("--text-column", default="text")
    train.add_argument("--dataset-config")
    train.add_argument("--revision")
    train.add_argument("--split", default="train")
    train.add_argument("--max-documents", type=int, default=10000)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--device")
    train.add_argument("--vocab-size", type=int, default=512)
    train.add_argument("--max-steps", type=int, default=1000)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--stop-after", type=int)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--context-length", type=int, default=128)
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--num-heads", type=int, default=4)
    train.add_argument("--num-layers", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--eval-interval", type=int, default=100)
    train.add_argument("--eval-batches", type=int, default=10)
    train.add_argument("--checkpoint-interval", type=int, default=500)
    generate = commands.add_parser("generate", help="generate from a V2 checkpoint")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--device")
    generate.add_argument("--max-new-tokens", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--greedy", action="store_true")
    return root


def main():
    args = parser().parse_args()
    device = get_device(args.device)
    if args.command == "train":
        documents = (stream_local(args.data, max_documents=args.max_documents) if args.data else
                     stream_huggingface(args.dataset, split=args.split, text_column=args.text_column,
                                        config=args.dataset_config, revision=args.revision,
                                        max_documents=args.max_documents))
        if args.resume:
            config = V2Config.from_dict(read_checkpoint(args.resume)["config"])
        else:
            config = V2Config(max_steps=args.max_steps, warmup_steps=args.warmup_steps,
                              batch_size=args.batch_size, context_length=args.context_length,
                              embedding_dim=args.embedding_dim, num_heads=args.num_heads,
                              num_layers=args.num_layers, learning_rate=args.learning_rate,
                              precision=args.precision, gradient_accumulation_steps=args.gradient_accumulation_steps,
                              eval_interval=args.eval_interval, eval_batches=args.eval_batches,
                              checkpoint_interval=args.checkpoint_interval)
        result = train_documents(documents, config=config, output_path=args.output, vocab_size=args.vocab_size,
                                 device=device, resume_path=args.resume, stop_after=args.stop_after)
        print(f"saved={result['checkpoint']} step={result['step']}")
    else:
        state = read_checkpoint(args.checkpoint)
        tokenizer = KoreanBPETokenizer.from_dict(state["tokenizer"])
        model = build_model(V2Config.from_dict(state["config"])).to(device)
        model.load_state_dict(state["model"])
        print(generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_k=args.top_k, greedy=args.greedy))


if __name__ == "__main__":
    main()
