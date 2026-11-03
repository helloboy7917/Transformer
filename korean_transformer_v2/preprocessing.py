"""Normalize, deduplicate, and split documents without train/validation leakage."""

import hashlib
import unicodedata
from collections.abc import Iterable


def normalize_document(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("documents must be strings")
    return " ".join(unicodedata.normalize("NFC", text).split())


def prepare_documents(documents: Iterable[str], *, validation_fraction: float = 0.1,
                      seed: int = 1337) -> tuple[list[str], list[str]]:
    """Return stable document-level splits; duplicate normalized documents are removed."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    train, validation, seen = [], [], set()
    threshold = int(validation_fraction * (1 << 256))
    for text in documents:
        text = normalize_document(text)
        if not text or text in seen:
            continue
        seen.add(text)
        digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
        (validation if int.from_bytes(digest, "big") < threshold else train).append(text)
    if not train or not validation:
        raise ValueError("both splits need documents; add data or adjust the seed or fraction")
    return train, validation


def encode_documents(documents: Iterable[str], tokenizer) -> list[int]:
    """Join documents with explicit end markers for causal language modelling."""
    tokens = []
    for document in documents:
        tokens.extend(tokenizer.encode(document, add_eos=True))
    return tokens
