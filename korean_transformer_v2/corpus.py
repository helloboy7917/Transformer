"""Disk-backed, deterministic corpus preparation with bounded Python memory."""

import hashlib
import json
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

import torch

from .preprocessing import normalize_document


class DiskCorpus:
    def __init__(self, documents, directory, *, validation_fraction=0.1, seed=1337):
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "documents.sqlite")
        self.db.execute("DROP TABLE IF EXISTS documents")
        self.db.execute("CREATE TABLE documents (digest TEXT PRIMARY KEY, split TEXT, text TEXT)")
        with self.db:
            for document in documents:
                text = normalize_document(document)
                if not text:
                    continue
                digest = hashlib.sha256(f"{seed}\0{text}".encode()).hexdigest()
                split = "validation" if int(digest, 16) < validation_fraction * 2**256 else "train"
                self.db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?, ?)", (digest, split, text))
        for split in ("train", "validation"):
            if not self.db.execute("SELECT 1 FROM documents WHERE split=? LIMIT 1", (split,)).fetchone():
                self.close()
                raise ValueError("both splits need documents")
        # Match the original V2 JSON fingerprint exactly, without building lists.
        fingerprint = hashlib.sha256(b"[")
        for index, split in enumerate(("train", "validation")):
            fingerprint.update(b",[" if index else b"[")
            for number, text in enumerate(self.documents(split)):
                if number:
                    fingerprint.update(b",")
                fingerprint.update(json.dumps(text, ensure_ascii=False, separators=(",", ":")).encode())
            fingerprint.update(b"]")
        fingerprint.update(b"]")
        self.fingerprint = fingerprint.hexdigest()

    def documents(self, split="train", limit=None):
        query = "SELECT text FROM documents WHERE split=? ORDER BY rowid"
        # tokenizers may consume this iterator on its Rust worker thread.
        connection = sqlite3.connect(self.directory / "documents.sqlite")
        try:
            rows = connection.execute(query + (" LIMIT ?" if limit else ""), (split, limit) if limit else (split,))
            for (text,) in rows:
                yield text
        finally:
            connection.close()

    def encode(self, tokenizer, split):
        path = self.directory / f"{split}.bin"
        count = 0
        with path.open("wb") as output:
            for document in self.documents(split):
                ids = tokenizer.encode(document, add_eos=True)
                output.write(struct.pack(f"<{len(ids)}q", *ids))
                count += len(ids)
        # File mapping keeps tokens on disk; only sampled microbatches move to CUDA.
        return torch.from_file(str(path), shared=False, size=count, dtype=torch.long)

    def write_tokenizer_sample(self, path, *, max_documents, max_characters):
        """Finish all SQLite reads on the caller thread before Rust training."""
        if max_documents <= 0 or max_characters <= 0:
            raise ValueError("tokenizer sample limits must be positive")
        remaining = max_characters
        with Path(path).open("x", encoding="utf-8", newline="\n") as output:
            with closing(self.documents(limit=max_documents)) as documents:
                for document in documents:
                    sample = document[:remaining]
                    output.write(sample + "\n")
                    remaining -= len(sample)
                    if remaining == 0:
                        break

    def close(self):
        self.db.close()


def save_prepared(directory, *, fingerprint, tokenizer, config, train_tokens, val_tokens):
    """Publish metadata only after both token files have finished writing."""
    manifest = {"format": "v2_prepared_1", "fingerprint": fingerprint,
                "tokenizer": tokenizer.to_dict(), "seed": config.seed,
                "validation_fraction": config.validation_fraction,
                "train_count": len(train_tokens), "validation_count": len(val_tokens)}
    path = Path(directory) / "prepared.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False)


def load_prepared(directory, config):
    from .tokenizer import KoreanBPETokenizer
    directory = Path(directory)
    manifest = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "v2_prepared_1" or manifest["seed"] != config.seed
            or manifest["validation_fraction"] != config.validation_fraction):
        raise ValueError("prepared corpus configuration mismatch")
    tokens = []
    for split in ("train", "validation"):
        count = manifest[f"{split}_count"]
        path = directory / f"{split}.bin"
        if count <= config.context_length or path.stat().st_size != count * 8:
            raise ValueError("prepared corpus token file is truncated or too short")
        tokens.append(torch.from_file(str(path), shared=False, size=count, dtype=torch.long))
    return manifest["fingerprint"], KoreanBPETokenizer.from_dict(manifest["tokenizer"]), *tokens
