import json

import pytest

from korean_transformer_v2.tokenizer import KoreanBPETokenizer


def test_bpe_roundtrips_korean_unicode_and_unseen_characters():
    tokenizer = KoreanBPETokenizer.train(["안녕하세요 안녕하세요", "한국어 학습"], vocab_size=280)
    text = "안녕하세요! 처음 보는 글자 🐈\n"
    tokens = tokenizer.encode(text, add_eos=True)
    assert tokens[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(tokens) == text
    assert len(tokenizer.encode("안녕하세요")) < len("안녕하세요".encode())


def test_bpe_is_deterministic_and_serializable():
    documents = ["가나다 가나다", "가나다라마바사"]
    first = KoreanBPETokenizer.train(documents, vocab_size=275)
    second = KoreanBPETokenizer.train(iter(documents), vocab_size=275)
    restored = KoreanBPETokenizer.from_dict(json.loads(json.dumps(first.to_dict())))
    assert first.to_dict() == second.to_dict() == restored.to_dict()
    assert restored.encode(documents[0]) == first.encode(documents[0])


def test_bpe_validation_and_document_boundaries():
    with pytest.raises(ValueError):
        KoreanBPETokenizer.train([""], 300)
    with pytest.raises(ValueError):
        KoreanBPETokenizer.train(["text"], 256)
    with pytest.raises(ValueError):
        KoreanBPETokenizer([(1, 999)])
    tokenizer = KoreanBPETokenizer.train(["a", "b"], 258, min_frequency=1)
    assert tokenizer.vocab_size == 257
    with pytest.raises(ValueError):
        tokenizer.decode([999])
