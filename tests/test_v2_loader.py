import sys
import types

import pytest

from korean_transformer_v2.data_loader import stream_huggingface, stream_local


def test_local_stream_is_lazy_and_bounded(tmp_path):
    path = tmp_path / "korean.txt"
    source = stream_local(path, max_documents=2)
    path.write_text("안녕\n\n세계\n세번째\n", encoding="utf-8")
    assert list(source) == ["안녕", "세계"]
    with pytest.raises(ValueError):
        list(stream_local(path, max_documents=0))


def test_huggingface_streams_and_checks_records(monkeypatch):
    calls = []
    def load_dataset(repo, **kwargs):
        calls.append((repo, kwargs))
        return iter([{"body": " 한국어 "}, {"body": ""}, {"body": "두번째"}])
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    records = stream_huggingface("org/corpus", text_column="body", max_documents=1, revision="abc")
    assert not calls
    assert list(records) == ["한국어"]
    assert calls[0][1]["streaming"] is True
    assert calls[0][1]["revision"] == "abc"
    with pytest.raises(ValueError, match="missing"):
        list(stream_huggingface("org/corpus"))
