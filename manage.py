"""Create, train, inspect, clone, and sample independent local GPT models."""

import argparse
import json
from pathlib import Path
import sys

from config import GPTConfig
from services.dataset_loader import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_SAMPLES,
    DatasetLoadError,
    DatasetSpec,
)
from services.model_manager import DEFAULT_MODELS_DIR, ModelManager, ModelManagerError


def _add_model_store_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="directory for independent model artifacts (default: trained_models)",
    )


def _add_source_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--dataset", help="Hugging Face Dataset Repository ID")
    group.add_argument("--data", type=Path, help="local UTF-8 .txt file")
    parser.add_argument("--dataset-config", help="Hugging Face dataset configuration name")
    parser.add_argument("--split", default=None, help="training split name (default: train)")
    parser.add_argument("--validation-split", default=None, help="official validation split name")
    parser.add_argument(
        "--data-file",
        action="append",
        help="optional file path or glob inside the Hub repository; may be repeated",
    )
    parser.add_argument("--text-column", help="text field for JSONL, CSV, or Parquet data")
    parser.add_argument("--revision", help="Hub branch, tag, or commit SHA (default: main)")
    parser.add_argument("--max-samples", type=int, default=None, help="maximum documents per split")
    parser.add_argument("--max-chars", type=int, default=None, help="maximum characters per split")
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)


def _dataset_spec(args, manager: ModelManager, *, defaults: dict | None = None) -> DatasetSpec | None:
    if not getattr(args, "dataset", None) and not getattr(args, "data", None):
        related_options = (
            "dataset_config",
            "split",
            "validation_split",
            "data_file",
            "text_column",
            "revision",
            "max_samples",
            "max_chars",
            "validation_fraction",
            "seed",
            "cache_dir",
        )
        if any(getattr(args, key, None) is not None for key in related_options):
            raise ModelManagerError("dataset options require either --dataset or --data")
        return None

    defaults = defaults or {}
    validation_fraction = getattr(args, "validation_fraction", None)
    if validation_fraction is None:
        validation_fraction = defaults.get("validation_fraction", GPTConfig().validation_fraction)
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = defaults.get("seed", GPTConfig().seed)
    source_type = "huggingface" if args.dataset else "local"
    return DatasetSpec(
        source_type=source_type,
        data_path=str(args.data.expanduser().resolve()) if args.data else None,
        repo_id=args.dataset,
        dataset_config=args.dataset_config,
        train_split=args.split or "train",
        validation_split=args.validation_split,
        require_validation_split=args.validation_split is not None,
        data_files=args.data_file,
        text_column=args.text_column,
        revision=args.revision,
        max_samples=args.max_samples if args.max_samples is not None else DEFAULT_MAX_SAMPLES,
        max_chars=args.max_chars if args.max_chars is not None else DEFAULT_MAX_CHARS,
        validation_fraction=validation_fraction,
        seed=seed,
        cache_dir=str((args.cache_dir or manager.cache_dir).expanduser().resolve()),
    )


def _add_create_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--feedforward-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create and initialize a model")
    _add_model_store_option(create)
    create.add_argument("--name", required=True)
    _add_source_arguments(create, required=True)
    _add_create_config_arguments(create)

    train = subparsers.add_parser("train", help="train or resume a named model")
    _add_model_store_option(train)
    train.add_argument("--name", required=True)
    train.add_argument("--max-steps", type=int, default=None, help="target total step, not extra steps")
    train.add_argument("--device", default=None, help="cpu, cuda, or a specific torch device")
    _add_source_arguments(train, required=False)

    listing = subparsers.add_parser("list", help="list available models")
    _add_model_store_option(listing)

    info = subparsers.add_parser("info", help="show configuration, dataset, log, and checkpoints")
    _add_model_store_option(info)
    info.add_argument("--name", required=True)

    generate = subparsers.add_parser("generate", help="generate text with a trained named model")
    _add_model_store_option(generate)
    generate.add_argument("--name", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--greedy", action="store_true")
    generate.add_argument("--device", default=None)

    clone = subparsers.add_parser("clone", help="clone model artifacts to a new independent model")
    _add_model_store_option(clone)
    clone.add_argument("--source", required=True)
    clone.add_argument("--name", required=True)
    return parser


def _print_model_list(manager: ModelManager) -> None:
    models = manager.list_models()
    if not models:
        print(f"No models found in {manager.models_dir}")
        return
    print(f"{'NAME':<24} {'STATUS':<14} {'DATASET':<32} {'STEP':>9} {'VAL LOSS':>12} {'PARAMETERS':>13}")
    for model in models:
        validation_loss = model["validation_loss"]
        validation_text = f"{validation_loss:.4f}" if isinstance(validation_loss, (float, int)) else "-"
        print(
            f"{model['name']:<24} {model['status']:<14} {str(model['dataset'])[:32]:<32} "
            f"{model['last_step']:>9} {validation_text:>12} {str(model['parameter_count'] or '-'):>13}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = ModelManager(models_dir=args.models_dir)
    try:
        if args.command == "create":
            validation_fraction = (
                args.validation_fraction
                if args.validation_fraction is not None
                else GPTConfig().validation_fraction
            )
            seed = args.seed if args.seed is not None else GPTConfig().seed
            spec = _dataset_spec(
                args,
                manager,
                defaults={"validation_fraction": validation_fraction, "seed": seed},
            )
            overrides = {
                key: value
                for key, value in {
                    "embedding_dim": args.embedding_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "context_length": args.context_length,
                    "feedforward_dim": args.feedforward_dim,
                    "dropout": args.dropout,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "max_steps": args.max_steps,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                    "eval_interval": args.eval_interval,
                    "eval_batches": args.eval_batches,
                    "checkpoint_interval": args.checkpoint_interval,
                    "validation_fraction": validation_fraction,
                    "seed": seed,
                }.items()
                if value is not None
            }
            result = manager.create(args.name, spec, overrides)
            print(
                f"Created model '{args.name}' | status: {result['status']} | "
                f"vocabulary: {result['tokenizer_vocabulary_size']} | "
                f"parameters: {result['parameter_count']:,}"
            )
            print(f"Model directory: {result['directory']}")
        elif args.command == "train":
            model_info = manager.info(args.name)
            spec = _dataset_spec(args, manager, defaults=model_info["config"])
            result = manager.train(
                args.name,
                max_steps=args.max_steps,
                device=args.device,
                dataset_override=spec,
            )
            print(
                f"Training state: {result['status']} | total step: {result['trained_steps']} | "
                f"best validation loss: {result['best_validation_loss']}"
            )
        elif args.command == "list":
            _print_model_list(manager)
        elif args.command == "info":
            print(json.dumps(manager.info(args.name), ensure_ascii=False, indent=2))
        elif args.command == "generate":
            print(
                manager.generate(
                    args.name,
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    greedy=args.greedy,
                    device=args.device,
                )
            )
        elif args.command == "clone":
            result = manager.clone(args.source, args.name)
            print(
                f"Cloned '{args.source}' to '{args.name}' | status: {result['status']} | "
                f"parameters: {result['parameter_count']:,}"
            )
            print(f"Model directory: {result['directory']}")
        return 0
    except (DatasetLoadError, ModelManagerError, FileExistsError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
