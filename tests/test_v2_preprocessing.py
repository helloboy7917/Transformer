import pytest

from korean_transformer_v2.preprocessing import encode_documents, normalize_document, prepare_documents
from korean_transformer_v2.tokenizer import KoreanBPETokenizer


def test_normalize_deduplicate_and_split_without_leakage():
    docs = [f"문서 {n}" for n in range(100)] + ["가", "가", " \n "]
    train, val = prepare_documents(docs, validation_fraction=0.3, seed=7)
    assert not set(train) & set(val)
    assert len(train) + len(val) == 101
    assert "가" in train + val
    train2, val2 = prepare_documents(reversed(docs), validation_fraction=0.3, seed=7)
    assert set(train) == set(train2)
    assert set(val) == set(val2)
    assert normalize_document("  안녕\n 세계  ") == "안녕 세계"


def test_split_validation_and_end_markers():
    with pytest.raises(ValueError):
        prepare_documents(["하나"], validation_fraction=0.1)
    with pytest.raises(ValueError):
        prepare_documents(["하나"], validation_fraction=1)
    tokenizer = KoreanBPETokenizer.train(["가", "나"], vocab_size=257)
    assert encode_documents(["가", "나"], tokenizer) == tokenizer.encode("가", add_eos=True) + tokenizer.encode("나", add_eos=True)
