from types import SimpleNamespace

import pytest

import services.dataset_loader as loader
from services.dataset_loader import DatasetLoadError, DatasetSpec, load_dataset_bundle


class StringFeature:
    dtype = "string"


class FakeSplit:
    def __init__(self, rows, features=None):
        self.rows = rows
        self.features = features or {"text": StringFeature()}

    def __iter__(self):
        yield from self.rows


def test_local_text_is_deduplicated_and_split_by_document(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(
        "First document about cats.\n\nSecond document about birds.\n\n"
        "Third document about trees.\n\nSecond document about birds.\n\n"
        "Fourth document about stars.\n",
        encoding="utf-8",
    )
    bundle = load_dataset_bundle(
        DatasetSpec(
            source_type="local",
            data_path=str(path),
            max_samples=None,
            max_chars=None,
            validation_fraction=0.25,
            seed=9,
        )
    )
    train = set(bundle.train_documents)
    validation = set(bundle.validation_documents)
    assert train.isdisjoint(validation)
    assert len(train) + len(validation) == 4
    assert bundle.metadata["duplicates_removed"] == 1
    assert bundle.metadata["revision"].startswith("sha256:")


def test_huggingface_loader_streams_selected_splits_and_records_commit(tmp_path, monkeypatch):
    calls = {}

    def fake_load_dataset(**kwargs):
        calls.update(kwargs)
        return {
            "train": FakeSplit([{"text": "alpha alpha alpha"}, {"text": "beta beta beta"}]),
            "validation": FakeSplit([{"text": "gamma gamma gamma"}]),
        }

    class FakeApi:
        def dataset_info(self, repo_id, *, revision, token):
            calls["info"] = (repo_id, revision, token)
            return SimpleNamespace(sha="abc123commit")

    monkeypatch.setattr(loader, "_get_hf_clients", lambda: (fake_load_dataset, FakeApi))
    spec = DatasetSpec(
        source_type="huggingface",
        repo_id="writer/example-corpus",
        revision="release-v1",
        max_samples=20,
        max_chars=1000,
        validation_fraction=0.2,
        cache_dir=str(tmp_path / "dataset-cache"),
    )
    bundle = load_dataset_bundle(spec)
    assert bundle.metadata["commit_sha"] == "abc123commit"
    assert bundle.metadata["revision"] == "abc123commit"
    assert bundle.metadata["validation_split"] == "validation"
    assert calls["info"] == ("writer/example-corpus", "release-v1", None)
    assert calls["streaming"] is True
    assert calls["revision"] == "abc123commit"
    assert calls["token"] is None
    assert len(bundle.validation_documents) == 1


def test_huggingface_loader_reports_private_dataset_auth_error(monkeypatch):
    class GatedRepoError(Exception):
        pass

    class FakeApi:
        def dataset_info(self, *args, **kwargs):
            raise GatedRepoError("access denied")

    monkeypatch.setattr(loader, "_get_hf_clients", lambda: (lambda **kwargs: None, FakeApi))
    spec = DatasetSpec(source_type="huggingface", repo_id="private/locked")
    with pytest.raises(DatasetLoadError, match="hf auth login"):
        load_dataset_bundle(spec)


def test_huggingface_loader_requires_text_column_when_ambiguous(monkeypatch):
    class FakeApi:
        def dataset_info(self, *args, **kwargs):
            return SimpleNamespace(sha="sha-for-test")

    def fake_load_dataset(**kwargs):
        features = {"title": StringFeature(), "body": StringFeature()}
        return {
            "train": FakeSplit([{"title": "one", "body": "alpha"}, {"title": "two", "body": "beta"}], features),
            "validation": FakeSplit([{"title": "three", "body": "gamma"}], features),
        }

    monkeypatch.setattr(loader, "_get_hf_clients", lambda: (fake_load_dataset, FakeApi))
    with pytest.raises(DatasetLoadError, match="--text-column"):
        load_dataset_bundle(DatasetSpec(source_type="huggingface", repo_id="public/ambiguous"))


def test_installed_datasets_streaming_rows_work_with_text_extractor(tmp_path):
    from datasets import load_dataset

    path = tmp_path / "rows.jsonl"
    path.write_text('{"text":"first paragraph"}\n{"text":"second paragraph"}\n', encoding="utf-8")
    stream = load_dataset(
        "json",
        data_files={"train": str(path)},
        split="train",
        streaming=True,
    )
    spec = DatasetSpec(source_type="huggingface", repo_id="local/mock", max_samples=1, max_chars=100)
    documents, column = loader._iter_split_documents(stream, spec, "train")
    assert documents == ["first paragraph"]
    assert column == "text"
