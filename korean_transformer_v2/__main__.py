"""Train and generate text with Korean Transformer V2."""

import argparse
from dataclasses import replace
from pathlib import Path

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
    source.add_argument("--prepared-cache", help="reuse prepared.json and token files without reading source")
    train.add_argument("--text-column", default="text")
    train.add_argument("--dataset-config")
    train.add_argument("--revision")
    train.add_argument("--split", default="train")
    limit = train.add_mutually_exclusive_group()
    limit.add_argument("--max-documents", type=int, default=10000)
    limit.add_argument("--all-documents", action="store_true", help="prepare the entire stream on disk before training")
    train.add_argument("--cache-dir", default="dataset_cache/v2")
    train.add_argument("--tokenizer-backend", choices=["fast", "python"], default="fast")
    train.add_argument("--tokenizer-file", help="reuse an existing tokenizer JSON")
    train.add_argument("--export-tokenizer", help="save tokenizer JSON to a new file")
    train.add_argument("--tokenizer-max-documents", type=int, default=20000)
    train.add_argument("--tokenizer-max-characters", type=int, default=20000000)
    train.add_argument("--log-path")
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--device")
    train.add_argument("--vocab-size", type=int, default=8000)
    train.add_argument("--max-steps", type=int, default=None, help="absolute target; omitted on resume keeps saved target")
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--stop-after", type=int)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--context-length", type=int, default=512)
    train.add_argument("--embedding-dim", type=int, default=256)
    train.add_argument("--num-heads", type=int, default=8)
    train.add_argument("--num-layers", type=int, default=6)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--validation-fraction", type=float, default=0.1)
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
        if args.export_tokenizer and Path(args.export_tokenizer).exists():
            raise FileExistsError("export tokenizer already exists")
        maximum = None if args.all_documents else args.max_documents
        documents = (stream_local(args.data, max_documents=maximum) if args.data else
                     stream_huggingface(args.dataset, split=args.split, text_column=args.text_column,
                                        config=args.dataset_config, revision=args.revision,
                                        max_documents=maximum))
        if args.prepared_cache:
            documents = None
        if args.resume:
            config = V2Config.from_dict(read_checkpoint(args.resume)["config"])
            if args.max_steps is not None:
                config = replace(config, max_steps=args.max_steps)
        else:
            config = V2Config(max_steps=args.max_steps if args.max_steps is not None else 1000, warmup_steps=args.warmup_steps,
                              batch_size=args.batch_size, context_length=args.context_length,
                              embedding_dim=args.embedding_dim, num_heads=args.num_heads,
                              num_layers=args.num_layers, learning_rate=args.learning_rate,
                              precision=args.precision, gradient_accumulation_steps=args.gradient_accumulation_steps,
                              grad_clip=args.grad_clip, weight_decay=args.weight_decay, seed=args.seed,
                              validation_fraction=args.validation_fraction,
                              eval_interval=args.eval_interval, eval_batches=args.eval_batches,
                              checkpoint_interval=args.checkpoint_interval)
        result = train_documents(documents, config=config, output_path=args.output, vocab_size=args.vocab_size,
                                 device=device, resume_path=args.resume, stop_after=args.stop_after,
                                 cache_dir=args.cache_dir, tokenizer_backend=args.tokenizer_backend,
                                 tokenizer_path=args.tokenizer_file, tokenizer_max_documents=args.tokenizer_max_documents,
                                 tokenizer_max_characters=args.tokenizer_max_characters, log_path=args.log_path,
                                 prepared_cache=args.prepared_cache)
        if args.export_tokenizer:
            KoreanBPETokenizer.from_dict(read_checkpoint(args.output)["tokenizer"]).save(args.export_tokenizer)
        print(f"saved={result['checkpoint']} step={result['step']}")
        print(f"prepared_cache={result['cache']} metrics={result['metrics']}")
    else:
        state = read_checkpoint(args.checkpoint)
        tokenizer = KoreanBPETokenizer.from_dict(state["tokenizer"])
        model = build_model(V2Config.from_dict(state["config"])).to(device)
        model.load_state_dict(state["model"])
        print(generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_k=args.top_k, greedy=args.greedy))


if __name__ == "__main__":
    main()
