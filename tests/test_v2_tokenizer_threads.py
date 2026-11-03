"""Regression coverage for SQLite iterators crossing the Rust BPE boundary."""

from contextlib import closing
import itertools
import sqlite3
import threading

import pytest

from korean_transformer_v2.corpus import DiskCorpus
from korean_transformer_v2.model import V2Config
from korean_transformer_v2.pipeline import train_documents
from korean_transformer_v2.tokenizer import FastBPETokenizer


def test_fast_training_accepts_cursor_created_on_calling_thread(tmp_path):
    # A cursor created before train_from_iterator cannot move to a Rust worker.
    with closing(sqlite3.connect(tmp_path / "source.sqlite")) as connection:
        connection.execute("CREATE TABLE source (text TEXT)")
        connection.executemany("INSERT INTO source VALUES (?)", [(f"한국어 학습 문서 {i}",) for i in range(300)])
        cursor = connection.execute("SELECT text FROM source ORDER BY rowid")
        owner = threading.get_ident()
        def documents():
            for (text,) in cursor:
                assert threading.get_ident() == owner
                yield text
        iterator = documents()
        first = next(iterator)  # The generator is already active on this thread.
        tokenizer = FastBPETokenizer.train(itertools.chain([first], iterator), vocab_size=300)
        text = "한국어 새로운 문자 🦊"
        assert tokenizer.decode(tokenizer.encode(text, add_eos=True)) == text


@pytest.mark.parametrize("document_limit,character_limit", [(2, 10000), (100, 7)])
def test_sample_file_is_bounded_and_closes_sqlite_iterator(tmp_path, monkeypatch, document_limit, character_limit):
    corpus = DiskCorpus((f"한국어 문서 번호 {i}" for i in range(100)), tmp_path / "corpus")
    try:
        expected = []
        remaining = character_limit
        for text in corpus.documents(limit=document_limit):
            expected.append(text[:remaining])
            remaining -= len(expected[-1])
            if not remaining:
                break
        original = corpus.documents
        closed = []
        def tracked(*args, **kwargs):
            with closing(original(*args, **kwargs)) as rows:
                try:
                    yield from rows
                finally:
                    closed.append(threading.get_ident())
        monkeypatch.setattr(corpus, "documents", tracked)
        path = tmp_path / "sample.txt"
        corpus.write_tokenizer_sample(path, max_documents=document_limit, max_characters=character_limit)
        assert closed == [threading.get_ident()]
        assert path.read_text(encoding="utf-8").splitlines() == expected
        assert sum(map(len, expected)) <= character_limit
        assert len(expected) <= document_limit
    finally:
        corpus.close()
    # The completed sample remains usable after the database is closed.
    assert FastBPETokenizer.train_file(path, vocab_size=300).vocab_size >= 257


def test_pipeline_finishes_sqlite_sample_before_rust_training(tmp_path, monkeypatch):
    owner = threading.get_ident()
    active = []
    original_documents = DiskCorpus.documents
    original_train = FastBPETokenizer.train_file.__func__
    def tracked(self, *args, **kwargs):
        assert threading.get_ident() == owner
        active.append(True)
        try:
            with closing(original_documents(self, *args, **kwargs)) as rows:
                for row in rows:
                    assert threading.get_ident() == owner
                    yield row
        finally:
            active.pop()
    calls = []
    def checked_train(cls, path, **kwargs):
        assert not active, "SQLite iterator must be closed before entering Rust"
        assert threading.get_ident() == owner
        calls.append(path)
        return original_train(cls, path, **kwargs)
    monkeypatch.setattr(DiskCorpus, "documents", tracked)
    monkeypatch.setattr(FastBPETokenizer, "train_file", classmethod(checked_train))
    config = V2Config(embedding_dim=8, num_heads=2, num_layers=1, context_length=4,
                      max_steps=1, warmup_steps=0, batch_size=2, eval_batches=1)
    result = train_documents((f"한국어 코퍼스 문서 번호 {i}" for i in range(100)), config=config,
                             output_path=tmp_path / "model.pt", tokenizer_backend="fast",
                             vocab_size=300, tokenizer_max_documents=3, tokenizer_max_characters=25)
    assert result["step"] == 1 and len(calls) == 1
    lines = calls[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 3 and sum(map(len, lines)) <= 25
    assert not active
