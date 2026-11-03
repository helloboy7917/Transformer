"""Disk-backed, deterministic corpus preparation with bounded Python memory."""

import hashlib
import json
import sqlite3
import struct
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
        fingerprint = hashlib.sha256()
        with self.db:
            for document in documents:
                text = normalize_document(document)
                if not text:
                    continue
                digest = hashlib.sha256(f"{seed}\0{text}".encode()).hexdigest()
                split = "validation" if int(digest, 16) < validation_fraction * 2**256 else "train"
                cursor = self.db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?, ?)", (digest, split, text))
                if cursor.rowcount:
                    fingerprint.update(bytes.fromhex(digest))
        self.fingerprint = fingerprint.hexdigest()
        for split in ("train", "validation"):
            if not self.db.execute("SELECT 1 FROM documents WHERE split=? LIMIT 1", (split,)).fetchone():
                self.close()
                raise ValueError("both splits need documents")

    def documents(self, split="train", limit=None):
        query = "SELECT text FROM documents WHERE split=? ORDER BY rowid"
        rows = self.db.execute(query + (" LIMIT ?" if limit else ""), (split, limit) if limit else (split,))
        for (text,) in rows:
            yield text

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

    def close(self):
        self.db.close()
