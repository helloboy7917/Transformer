"""Streaming document sources for local UTF-8 files and Hugging Face datasets."""

from collections.abc import Iterator
from pathlib import Path


DATASET_PRESETS = {
    "korean-wikipedia": ("wikimedia/wikipedia", "20231101.ko"),
    "kormix": ("AdaMLLab/KorMix", None),
}


def stream_local(path: str | Path, *, max_documents: int | None = None) -> Iterator[str]:
    """Yield nonempty lines without loading the input file into memory."""
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    count = 0
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            document = line.strip()
            if document:
                yield document
                count += 1
                if max_documents is not None and count >= max_documents:
                    return


def stream_huggingface(repo_id: str, *, split: str = "train", text_column: str = "text",
                       config: str | None = None, revision: str | None = None,
                       max_documents: int | None = None) -> Iterator[str]:
    """Stream text records; authentication is delegated to the Hugging Face client."""
    if not repo_id.strip() or not split.strip() or not text_column.strip():
        raise ValueError("repo_id, split and text_column must be nonempty")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    from datasets import load_dataset

    if repo_id in DATASET_PRESETS:
        repo_id, default_config = DATASET_PRESETS[repo_id]
        config = config or default_config

    records = load_dataset(repo_id, name=config, split=split, revision=revision, streaming=True)
    count = 0
    for record in records:
        if text_column not in record:
            raise ValueError(f"dataset record is missing text column {text_column!r}")
        document = record[text_column]
        if not isinstance(document, str):
            raise ValueError(f"dataset column {text_column!r} must contain strings")
        document = document.strip()
        if document:
            yield document
            count += 1
            if max_documents is not None and count >= max_documents:
                return
