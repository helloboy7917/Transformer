"""Load bounded local or Hugging Face text documents for causal LM training."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import itertools
from pathlib import Path
import random
import unicodedata


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CACHE = PROJECT_DIR / "dataset_cache"
DEFAULT_MAX_SAMPLES = 10_000
DEFAULT_MAX_CHARS = 2_000_000


class DatasetLoadError(ValueError):
    """An actionable dataset selection, access, or data-format error."""


@dataclass
class DatasetSpec:
    source_type: str
    data_path: str | None = None
    repo_id: str | None = None
    dataset_config: str | None = None
    train_split: str = "train"
    validation_split: str | None = None
    require_validation_split: bool = False
    data_files: list[str] | None = None
    text_column: str | None = None
    revision: str | None = None
    max_samples: int | None = DEFAULT_MAX_SAMPLES
    max_chars: int | None = DEFAULT_MAX_CHARS
    validation_fraction: float = 0.1
    seed: int = 1337
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.source_type not in {"local", "huggingface"}:
            raise DatasetLoadError("source_type must be 'local' or 'huggingface'")
        if self.source_type == "local" and not self.data_path:
            raise DatasetLoadError("a local dataset requires data_path")
        if self.source_type == "huggingface" and not self.repo_id:
            raise DatasetLoadError("a Hugging Face dataset requires repo_id")
        if self.max_samples is not None and self.max_samples <= 0:
            raise DatasetLoadError("max_samples must be positive")
        if self.max_chars is not None and self.max_chars <= 0:
            raise DatasetLoadError("max_chars must be positive")
        if not 0 < self.validation_fraction < 1:
            raise DatasetLoadError("validation_fraction must be between 0 and 1")
        if not self.train_split.strip():
            raise DatasetLoadError("train_split cannot be empty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "DatasetSpec":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})


@dataclass
class DatasetBundle:
    train_documents: list[str]
    validation_documents: list[str]
    metadata: dict


def _document_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def _deduplicate(documents: list[str]) -> tuple[list[str], int]:
    unique = []
    seen = set()
    duplicate_count = 0
    for document in documents:
        cleaned = document.strip()
        key = _document_key(cleaned)
        if not key:
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique, duplicate_count


def _limit_documents(documents, max_samples: int | None, max_chars: int | None) -> list[str]:
    result = []
    used_chars = 0
    for document in documents:
        if max_samples is not None and len(result) >= max_samples:
            break
        if not isinstance(document, str):
            raise DatasetLoadError("each dataset document must contain text as a string")
        cleaned = document.strip()
        if not cleaned:
            continue
        if max_chars is not None:
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            if len(cleaned) > remaining:
                cleaned = cleaned[:remaining]
        if cleaned:
            result.append(cleaned)
            used_chars += len(cleaned)
    return result


def _split_documents(
    train_documents: list[str],
    validation_documents: list[str] | None,
    *,
    validation_fraction: float,
    seed: int,
    strict_validation: bool = False,
) -> tuple[list[str], list[str], int]:
    train_documents, train_duplicates = _deduplicate(train_documents)
    if len(train_documents) < 2:
        raise DatasetLoadError(
            "at least two distinct training documents are required for a document-level split"
        )

    if validation_documents is not None:
        validation_documents, val_duplicates = _deduplicate(validation_documents)
        train_keys = {_document_key(item) for item in train_documents}
        before = len(validation_documents)
        validation_documents = [
            item for item in validation_documents if _document_key(item) not in train_keys
        ]
        leaked = before - len(validation_documents)
        if validation_documents:
            return train_documents, validation_documents, train_duplicates + val_duplicates + leaked
        if strict_validation:
            raise DatasetLoadError(
                "the selected validation split is empty after duplicate and leakage checks"
            )
        # If a conventional validation split is absent or unusable, split the
        # selected training documents themselves, keeping whole documents intact.
    elif strict_validation:
        raise DatasetLoadError("the requested validation split was not found")

    rng = random.Random(seed)
    shuffled = list(train_documents)
    rng.shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_documents = shuffled[:validation_count]
    train_documents = shuffled[validation_count:]
    return train_documents, validation_documents, train_duplicates


def _fingerprint(train_documents: list[str], validation_documents: list[str]) -> str:
    digest = hashlib.sha256()
    for split_name, documents in (("train", train_documents), ("validation", validation_documents)):
        digest.update(split_name.encode("utf-8"))
        for document in documents:
            digest.update(b"\0")
            digest.update(document.encode("utf-8"))
    return digest.hexdigest()


def _bundle_metadata(spec: DatasetSpec, train_docs: list[str], val_docs: list[str], **extra) -> dict:
    metadata = {
        "source_type": spec.source_type,
        "source_spec": spec.to_dict(),
        "train_samples": len(train_docs),
        "validation_samples": len(val_docs),
        "train_characters": sum(map(len, train_docs)),
        "validation_characters": sum(map(len, val_docs)),
        "dataset_fingerprint": _fingerprint(train_docs, val_docs),
        "preprocessing": {
            "document_level_split": True,
            "deduplicate_normalized_documents": True,
            "validation_fraction": spec.validation_fraction,
            "seed": spec.seed,
            "max_samples_per_split": spec.max_samples,
            "max_characters_per_split": spec.max_chars,
            "text_column": spec.text_column,
        },
    }
    metadata.update(extra)
    return metadata


def _load_local(spec: DatasetSpec) -> DatasetBundle:
    path = Path(spec.data_path).expanduser().resolve()
    if not path.is_file():
        raise DatasetLoadError(f"local dataset file does not exist: {path}")
    if path.suffix.lower() != ".txt":
        raise DatasetLoadError("local datasets currently support UTF-8 .txt files")
    def paragraph_stream():
        paragraph = []
        with path.open("r", encoding="utf-8-sig", newline=None) as source:
            for line in source:
                if line.strip():
                    paragraph.append(line)
                elif paragraph:
                    yield "".join(paragraph).strip()
                    paragraph.clear()
            if paragraph:
                yield "".join(paragraph).strip()

    try:
        documents = _limit_documents(paragraph_stream(), spec.max_samples, spec.max_chars)
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(f"local dataset must be valid UTF-8: {path}") from exc
    if not documents:
        raise DatasetLoadError(f"local dataset is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    file_sha256 = digest.hexdigest()
    train_docs, val_docs, duplicates = _split_documents(
        documents,
        None,
        validation_fraction=spec.validation_fraction,
        seed=spec.seed,
    )
    metadata = _bundle_metadata(
        spec,
        train_docs,
        val_docs,
        path=str(path),
        format="txt",
        file_size_bytes=path.stat().st_size,
        file_sha256=file_sha256,
        revision=f"sha256:{file_sha256}",
        train_split="document_split",
        validation_split="document_split",
        duplicates_removed=duplicates,
    )
    return DatasetBundle(train_docs, val_docs, metadata)


def _get_hf_clients():
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise DatasetLoadError(
            "Hugging Face dataset support requires the 'datasets' package. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return load_dataset, HfApi


def _string_feature_columns(dataset) -> list[str]:
    features = getattr(dataset, "features", None)
    if not isinstance(features, Mapping):
        return []
    return [
        name
        for name, feature in features.items()
        if getattr(feature, "dtype", None) in {"string", "large_string"}
    ]


def _choose_text_column(dataset, first_row, requested: str | None) -> str:
    if not isinstance(first_row, Mapping):
        if isinstance(first_row, str) and requested is None:
            return "__row_text__"
        raise DatasetLoadError("dataset rows must be records with a text column")
    if requested:
        if requested not in first_row:
            raise DatasetLoadError(
                f"text column '{requested}' was not found; available columns: {', '.join(first_row)}"
            )
        if first_row[requested] is not None and not isinstance(first_row[requested], str):
            raise DatasetLoadError(f"text column '{requested}' must contain strings")
        return requested

    candidates = [
        name
        for name in _string_feature_columns(dataset)
        if name in first_row
    ]
    if "text" in first_row and first_row["text"] is None:
        return "text"
    if not candidates:
        candidates = [name for name, value in first_row.items() if isinstance(value, str)]
    if "text" in candidates:
        return "text"
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise DatasetLoadError("no string text column was found in the selected dataset split")
    raise DatasetLoadError(
        "multiple string columns were found ("
        + ", ".join(candidates)
        + "); choose the training field with --text-column"
    )


def _iter_split_documents(dataset, spec: DatasetSpec, split_name: str) -> tuple[list[str], str | None]:
    iterator = iter(dataset)
    try:
        first_row = next(iterator)
    except StopIteration:
        return [], spec.text_column
    selected_column = _choose_text_column(dataset, first_row, spec.text_column)
    rows = itertools.chain((first_row,), iterator)

    def text_values():
        for row in rows:
            if selected_column == "__row_text__":
                value = row
            elif isinstance(row, Mapping):
                if selected_column not in row:
                    raise DatasetLoadError(
                        f"row in split '{split_name}' is missing text column '{selected_column}'"
                    )
                value = row[selected_column]
            else:
                raise DatasetLoadError(f"rows in split '{split_name}' have inconsistent formats")
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise DatasetLoadError(
                    f"column '{selected_column}' in split '{split_name}' contains non-text values"
                )
            yield value

    documents = _limit_documents(text_values(), spec.max_samples, spec.max_chars)
    return documents, selected_column


def _friendly_hub_error(repo_id: str, exc: Exception) -> DatasetLoadError:
    kind = type(exc).__name__.lower()
    message = str(exc)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in {401, 403} or any(
        word in kind for word in ("gated", "repositorynotfound", "unauthorized", "forbidden")
    ):
        return DatasetLoadError(
            f"Could not access Hugging Face dataset '{repo_id}'. Check the repository ID, "
            "dataset permissions, and your existing login (run 'hf auth login' if needed). "
            "No access token is stored by this project."
        )
    if "feature" in kind or "builder" in kind or "datasetgeneration" in kind:
        return DatasetLoadError(
            f"Could not read the dataset format for '{repo_id}'. Supported repository data "
            "formats include text, JSONL, CSV, and Parquet. Check --data-file and --text-column."
        )
    return DatasetLoadError(f"Could not load Hugging Face dataset '{repo_id}': {message}")


def _load_huggingface(spec: DatasetSpec) -> DatasetBundle:
    load_dataset, HfApi = _get_hf_clients()
    requested_revision = spec.revision or "main"
    try:
        info = HfApi().dataset_info(spec.repo_id, revision=requested_revision, token=None)
        commit_sha = info.sha
        cache_dir = Path(spec.cache_dir or DEFAULT_DATASET_CACHE).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        split_datasets = load_dataset(
            path=spec.repo_id,
            name=spec.dataset_config,
            data_files=spec.data_files,
            revision=commit_sha,
            streaming=True,
            cache_dir=str(cache_dir),
            token=None,
        )
    except Exception as exc:
        raise _friendly_hub_error(spec.repo_id, exc) from exc

    if not isinstance(split_datasets, Mapping):
        raise DatasetLoadError(
            "Hugging Face loader did not return named dataset splits; choose a dataset with a train split"
        )
    split_names = list(split_datasets.keys())
    if spec.train_split not in split_datasets:
        raise DatasetLoadError(
            f"training split '{spec.train_split}' was not found in '{spec.repo_id}'. "
            f"Available splits: {', '.join(split_names)}"
        )

    try:
        train_docs, selected_column = _iter_split_documents(
            split_datasets[spec.train_split], spec, spec.train_split
        )
        val_split = spec.validation_split
        if val_split is None:
            val_split = next(
                (candidate for candidate in ("validation", "valid", "dev") if candidate in split_datasets),
                None,
            )
        if val_split is not None and val_split not in split_datasets:
            if spec.require_validation_split:
                raise DatasetLoadError(
                    f"validation split '{val_split}' was not found. Available splits: {', '.join(split_names)}"
                )
            val_split = None
        validation_docs = None
        if val_split is not None:
            validation_docs, validation_column = _iter_split_documents(
                split_datasets[val_split], spec, val_split
            )
            if selected_column and validation_column and selected_column != validation_column:
                raise DatasetLoadError(
                    f"training split uses text column '{selected_column}' but validation split "
                    f"uses '{validation_column}'; set --text-column to a shared field"
                )
        train_docs, val_docs, duplicates = _split_documents(
            train_docs,
            validation_docs,
            validation_fraction=spec.validation_fraction,
            seed=spec.seed,
            strict_validation=spec.require_validation_split,
        )
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise _friendly_hub_error(spec.repo_id, exc) from exc

    metadata = _bundle_metadata(
        spec,
        train_docs,
        val_docs,
        repo_id=spec.repo_id,
        dataset_config=spec.dataset_config,
        requested_revision=requested_revision,
        revision=commit_sha,
        commit_sha=commit_sha,
        available_splits=split_names,
        train_split=spec.train_split,
        validation_split=val_split or "document_split",
        data_files=spec.data_files,
        text_column=selected_column,
        cache_dir=str(cache_dir),
        duplicates_removed=duplicates,
    )
    return DatasetBundle(train_docs, val_docs, metadata)


def load_dataset_bundle(spec: DatasetSpec) -> DatasetBundle:
    """Load bounded text documents and create a leak-checked document split."""
    if spec.source_type == "local":
        return _load_local(spec)
    return _load_huggingface(spec)
