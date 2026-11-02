"""Lifecycle management for independent Transformer model instances."""

from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import torch

from config import GPTConfig, get_device
from data.tokenizer import CharTokenizer
from model.gpt import GPTLanguageModel, count_parameters
from services.dataset_loader import DatasetSpec, load_dataset_bundle
from services.trainer import set_seed, train_loop
from utils.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_DIR / "trained_models"
ARCHITECTURE_FIELDS = (
    "vocab_size",
    "embedding_dim",
    "num_heads",
    "num_layers",
    "context_length",
    "feedforward_dim",
    "dropout",
)


class ModelManagerError(RuntimeError):
    """A model lifecycle or compatibility error with a user-facing explanation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelManagerError(f"model metadata file is missing: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ModelManagerError(f"model metadata is invalid JSON: {path.name}") from exc


def _corpus_text(documents: list[str]) -> str:
    return "\n\n".join(document.strip() for document in documents if document.strip())


def _required_context(documents: list[str], context_length: int, label: str) -> str:
    text = _corpus_text(documents)
    if len(text) <= context_length:
        raise ModelManagerError(
            f"{label} split has {len(text)} characters, but context_length is {context_length}. "
            "Provide more documents or create a model with a shorter context length."
        )
    return text


def _architecture_from_checkpoint(checkpoint: dict) -> dict:
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ModelManagerError("checkpoint does not contain a model configuration")
    return {key: config.get(key) for key in ARCHITECTURE_FIELDS}


class ModelManager:
    def __init__(self, models_dir: str | Path | None = None, cache_dir: str | Path | None = None) -> None:
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR).expanduser().resolve()
        self.cache_dir = Path(cache_dir or (PROJECT_DIR / "dataset_cache")).expanduser().resolve()

    @staticmethod
    def validate_name(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name or ""):
            raise ModelManagerError(
                "model name must be 1-64 characters: letters, numbers, dot, underscore, or hyphen; "
                "it must start with a letter or number"
            )
        return name

    def model_dir(self, name: str) -> Path:
        name = self.validate_name(name)
        return self.models_dir / name

    def _ensure_model(self, name: str) -> Path:
        directory = self.model_dir(name)
        if not directory.is_dir():
            raise ModelManagerError(f"model '{name}' does not exist; create it first")
        return directory

    def _create_temp_dir(self, name: str) -> tuple[Path, Path]:
        target = self.model_dir(name)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"model '{name}' already exists; existing files were left unchanged")
        temporary = self.models_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        return temporary, target

    def create(
        self,
        name: str,
        dataset_spec: DatasetSpec,
        config_overrides: dict | None = None,
    ) -> dict:
        """Create an isolated model directory and save its random initial weights."""
        self.validate_name(name)
        target = self.model_dir(name)
        if target.exists():
            raise FileExistsError(f"model '{name}' already exists; existing files were left unchanged")

        bundle = load_dataset_bundle(dataset_spec)
        train_text = _required_context(bundle.train_documents, 1, "training")
        tokenizer = CharTokenizer.from_text(train_text)
        values = dict(config_overrides or {})
        values["vocab_size"] = tokenizer.vocab_size
        config = GPTConfig(**values)
        train_text = _required_context(bundle.train_documents, config.context_length, "training")
        validation_text = _required_context(
            bundle.validation_documents, config.context_length, "validation"
        )
        # Include the document separator in the vocabulary before initializing embeddings.
        if "\n" not in tokenizer.stoi:
            tokenizer = CharTokenizer.from_text(train_text + "\n\n")
            config.vocab_size = tokenizer.vocab_size

        set_seed(config.seed)
        model = GPTLanguageModel(config)
        temporary, target = self._create_temp_dir(name)
        try:
            _write_json(temporary / "config.json", config.to_dict())
            _write_json(temporary / "tokenizer.json", tokenizer.to_dict())
            _write_json(temporary / "dataset_info.json", bundle.metadata)
            _write_json(
                temporary / "model_state.json",
                {
                    "name": name,
                    "status": "initialized",
                    "created_at": _now(),
                    "updated_at": _now(),
                    "trained_steps": 0,
                    "best_validation_loss": None,
                    "parameter_count": count_parameters(model),
                },
            )
            save_checkpoint(
                temporary / "initial.pt",
                model,
                None,
                step=0,
                model_config=config.to_dict(),
                tokenizer=tokenizer.to_dict(),
                metadata={"dataset_fingerprint": bundle.metadata["dataset_fingerprint"]},
            )
            temporary.rename(target)
        except Exception:
            # Only the temporary directory created by this call is eligible for cleanup.
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.info(name)

    def list_models(self) -> list[dict]:
        if not self.models_dir.exists():
            return []
        results = []
        for directory in sorted(self.models_dir.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                state = _read_json(directory / "model_state.json")
                config = GPTConfig.from_dict(_read_json(directory / "config.json"))
                dataset_info = _read_json(directory / "dataset_info.json")
                dataset_label = (
                    dataset_info.get("repo_id")
                    or dataset_info.get("path")
                    or dataset_info.get("source_type", "unknown")
                )
                if dataset_info.get("source_type") == "local":
                    dataset_label = Path(dataset_label).name
                step = int(state.get("trained_steps", 0))
                validation_loss = state.get("best_validation_loss")
                for filename in ("best.pt", "last.pt"):
                    checkpoint_path = directory / filename
                    if checkpoint_path.is_file():
                        checkpoint = read_checkpoint(checkpoint_path, map_location="cpu")
                        step = max(step, int(checkpoint.get("step", 0)))
                        checkpoint_loss = checkpoint.get("best_val_loss")
                        if checkpoint_loss is not None and (
                            validation_loss is None or checkpoint_loss < validation_loss
                        ):
                            validation_loss = checkpoint_loss
                results.append(
                    {
                        "name": directory.name,
                        "dataset": dataset_label,
                        "status": state.get("status", "unknown"),
                        "parameter_count": state.get("parameter_count"),
                        "last_step": step,
                        "validation_loss": validation_loss,
                        "context_length": config.context_length,
                    }
                )
            except (ModelManagerError, OSError, ValueError, KeyError):
                results.append(
                    {
                        "name": directory.name,
                        "dataset": "unknown",
                        "status": "metadata-error",
                        "parameter_count": None,
                        "last_step": 0,
                        "validation_loss": None,
                    }
                )
        return results

    def info(self, name: str) -> dict:
        directory = self._ensure_model(name)
        config_values = _read_json(directory / "config.json")
        config = GPTConfig.from_dict(config_values)
        tokenizer = CharTokenizer.from_dict(_read_json(directory / "tokenizer.json"))
        dataset_info = _read_json(directory / "dataset_info.json")
        state = _read_json(directory / "model_state.json")
        last_step = int(state.get("trained_steps", 0))
        best_loss = state.get("best_validation_loss")
        checkpoint_files = []
        for filename in ("initial.pt", "best.pt", "last.pt"):
            path = directory / filename
            if path.is_file():
                checkpoint_files.append(filename)
                if filename == "last.pt":
                    saved = read_checkpoint(path, map_location="cpu")
                    last_step = max(last_step, int(saved.get("step", 0)))
                elif filename == "best.pt":
                    saved = read_checkpoint(path, map_location="cpu")
                    last_step = max(last_step, int(saved.get("step", 0)))
                    best_loss = saved.get("best_val_loss", best_loss)
        log_tail = []
        log_path = directory / "training_log.csv"
        if log_path.exists():
            with log_path.open("r", newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file))
            log_tail = rows[-5:]
        return {
            "name": name,
            "directory": str(directory),
            "status": state.get("status", "unknown"),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "trained_steps": last_step,
            "best_validation_loss": best_loss,
            "parameter_count": state.get("parameter_count"),
            "config": config.to_dict(),
            "tokenizer_vocabulary_size": tokenizer.vocab_size,
            "tokenizer_character_count": len(tokenizer.itos),
            "dataset": dataset_info,
            "checkpoints": checkpoint_files,
            "training_log_tail": log_tail,
        }

    def _read_model_files(self, name: str):
        directory = self._ensure_model(name)
        config = GPTConfig.from_dict(_read_json(directory / "config.json"))
        tokenizer = CharTokenizer.from_dict(_read_json(directory / "tokenizer.json"))
        dataset_info = _read_json(directory / "dataset_info.json")
        state = _read_json(directory / "model_state.json")
        return directory, config, tokenizer, dataset_info, state

    def _resolve_training_data(self, dataset_info: dict, override: DatasetSpec | None):
        if override is not None:
            return load_dataset_bundle(override), True
        try:
            spec = DatasetSpec.from_dict(dataset_info["source_spec"])
        except (KeyError, TypeError) as exc:
            raise ModelManagerError("model dataset metadata is missing its source specification") from exc
        if spec.source_type == "huggingface" and dataset_info.get("commit_sha"):
            spec.revision = dataset_info["commit_sha"]
        return load_dataset_bundle(spec), False

    def train(
        self,
        name: str,
        *,
        max_steps: int | None = None,
        device: str | None = None,
        dataset_override: DatasetSpec | None = None,
    ) -> dict:
        directory, config, tokenizer, dataset_info, state = self._read_model_files(name)
        if max_steps is not None:
            if max_steps < 0:
                raise ModelManagerError("max_steps must be non-negative")
            config.max_steps = max_steps
        bundle, is_override = self._resolve_training_data(dataset_info, dataset_override)
        if not is_override and bundle.metadata["dataset_fingerprint"] != dataset_info.get(
            "dataset_fingerprint"
        ):
            raise ModelManagerError(
                "loaded dataset content does not match the model's registered dataset version. "
                "The source may have changed; create a new model or pass a compatible dataset explicitly."
            )

        train_text = _required_context(bundle.train_documents, config.context_length, "training")
        validation_text = _required_context(
            bundle.validation_documents, config.context_length, "validation"
        )
        combined = train_text + "\n\n" + validation_text
        unknown = sorted(set(combined) - set(tokenizer.stoi))
        if is_override and unknown:
            preview = " ".join(repr(char) for char in unknown[:10])
            raise ModelManagerError(
                f"dataset is incompatible with model '{name}': {len(unknown)} character(s) "
                f"are outside its tokenizer vocabulary ({preview}). Create a new model with this dataset."
            )

        checkpoint = None
        checkpoint_path = None
        checkpoint_step = -1
        for filename in ("initial.pt", "best.pt", "last.pt"):
            candidate_path = directory / filename
            if not candidate_path.is_file():
                continue
            candidate = read_checkpoint(candidate_path, map_location="cpu")
            candidate_step = int(candidate.get("step", 0))
            if candidate_step >= checkpoint_step:
                checkpoint = candidate
                checkpoint_path = candidate_path
                checkpoint_step = candidate_step
        if checkpoint is None or checkpoint_path is None:
            raise ModelManagerError(f"model '{name}' has no initial checkpoint")
        expected_architecture = {key: config.to_dict().get(key) for key in ARCHITECTURE_FIELDS}
        if _architecture_from_checkpoint(checkpoint) != expected_architecture:
            raise ModelManagerError(
                "model architecture settings in config.json do not match the saved checkpoint. "
                "Architecture settings cannot be changed after model creation."
            )
        if checkpoint.get("tokenizer") != tokenizer.to_dict():
            raise ModelManagerError("saved checkpoint tokenizer does not match tokenizer.json")
        checkpoint_metadata = checkpoint.get("metadata") or {}
        if (
            not is_override
            and checkpoint_metadata.get("dataset_fingerprint")
            and checkpoint_metadata["dataset_fingerprint"] != bundle.metadata["dataset_fingerprint"]
        ):
            raise ModelManagerError(
                "checkpoint was trained with a different dataset fingerprint; refusing to mix datasets"
            )

        if max_steps is not None:
            _write_json(directory / "config.json", config.to_dict())
        target_device = get_device(device)
        model = GPTLanguageModel(config).to(target_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = checkpoint.get("best_val_loss")
        best_val_loss = best_val_loss if best_val_loss is not None else float("inf")

        train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long, device=target_device)
        validation_tokens = torch.tensor(
            tokenizer.encode(validation_text), dtype=torch.long, device=target_device
        )
        _write_json(
            directory / "model_state.json",
            {**state, "status": "training", "updated_at": _now(), "last_error": None},
        )
        set_seed(config.seed + start_step)
        try:
            result = train_loop(
                model,
                train_tokens,
                validation_tokens,
                config,
                tokenizer.to_dict(),
                directory,
                optimizer=optimizer,
                device=target_device,
                start_step=start_step,
                best_val_loss=best_val_loss,
                model_name=name,
                checkpoint_metadata={
                    "dataset_fingerprint": bundle.metadata["dataset_fingerprint"],
                    "dataset_revision": bundle.metadata.get("revision"),
                },
            )
        except Exception as exc:
            _write_json(
                directory / "model_state.json",
                {
                    **state,
                    "status": "interrupted",
                    "updated_at": _now(),
                    "trained_steps": start_step,
                    "best_validation_loss": None if best_val_loss == float("inf") else best_val_loss,
                    "last_error": str(exc),
                },
            )
            raise

        if is_override:
            history = list(dataset_info.get("fine_tuning_datasets", []))
            history.append(
                {
                    "added_at": _now(),
                    "previous_fingerprint": dataset_info.get("dataset_fingerprint"),
                    "dataset_fingerprint": bundle.metadata["dataset_fingerprint"],
                    "source": bundle.metadata.get("repo_id") or bundle.metadata.get("path"),
                    "revision": bundle.metadata.get("revision"),
                }
            )
            bundle.metadata["base_dataset"] = dataset_info.get("base_dataset", dataset_info)
            bundle.metadata["fine_tuning_datasets"] = history
            _write_json(directory / "dataset_info.json", bundle.metadata)
            dataset_info = bundle.metadata

        state = {
            **state,
            "status": "trained" if result["step"] > 0 else "initialized",
            "updated_at": _now(),
            "trained_steps": result["step"],
            "best_validation_loss": None
            if result["best_val_loss"] == float("inf")
            else result["best_val_loss"],
            "last_validation_loss": result["last_validation_loss"],
            "last_error": None,
        }
        _write_json(directory / "model_state.json", state)
        return self.info(name)

    def clone(self, source: str, name: str) -> dict:
        """Copy model artifacts into a new directory; source artifacts remain untouched."""
        self.validate_name(source)
        self.validate_name(name)
        source_dir = self._ensure_model(source)
        temporary, target = self._create_temp_dir(name)
        try:
            for filename in ("config.json", "tokenizer.json", "dataset_info.json", "initial.pt", "best.pt", "last.pt", "training_log.csv"):
                source_file = source_dir / filename
                if source_file.is_file():
                    shutil.copy2(source_file, temporary / filename)
            source_state = _read_json(source_dir / "model_state.json")
            cloned_state = {
                **source_state,
                "name": name,
                "status": "cloned",
                "created_at": _now(),
                "updated_at": _now(),
                "cloned_from": source,
                "source_status": source_state.get("status"),
            }
            _write_json(temporary / "model_state.json", cloned_state)
            cloned_dataset = _read_json(temporary / "dataset_info.json")
            cloned_dataset["cloned_from"] = source
            _write_json(temporary / "dataset_info.json", cloned_dataset)
            temporary.rename(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.info(name)

    def generate(
        self,
        name: str,
        prompt: str,
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
        device: str | None = None,
    ) -> str:
        directory, config, tokenizer, _, _ = self._read_model_files(name)
        checkpoint_path = directory / "best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = directory / "last.pt"
        if not checkpoint_path.exists():
            raise ModelManagerError(f"model '{name}' has not been trained; run 'manage.py train' first")
        checkpoint = read_checkpoint(checkpoint_path, map_location="cpu")
        if int(checkpoint.get("step", 0)) <= 0:
            raise ModelManagerError(f"model '{name}' has not been trained; run 'manage.py train' first")
        if _architecture_from_checkpoint(checkpoint) != {
            key: config.to_dict().get(key) for key in ARCHITECTURE_FIELDS
        }:
            raise ModelManagerError("model configuration and checkpoint architecture do not match")
        target_device = get_device(device)
        model = GPTLanguageModel(config).to(target_device)
        load_checkpoint(checkpoint_path, model, map_location=target_device)
        encoded = tokenizer.encode(prompt)
        if not encoded:
            encoded = [tokenizer.unk_token_id]
        tokens = torch.tensor([encoded], dtype=torch.long, device=target_device)
        generated = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        return tokenizer.decode(generated[0])
