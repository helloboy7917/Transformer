import pytest
from korean_transformer_v2.tokenizer import FastBPETokenizer, KoreanBPETokenizer


@pytest.mark.parametrize("size", [8000, 16000])
def test_fast_bpe_unicode_persistence_and_legacy_dispatch(tmp_path, size):
    tokenizer = FastBPETokenizer.train(iter(["한국어 학습 문서입니다." * 10, "안녕하세요 세계"]), size)
    text = "처음 보는 문자 🦊\n한국어"
    assert tokenizer.decode(tokenizer.encode(text, add_eos=True)) == text
    assert 257 <= tokenizer.vocab_size <= size
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = KoreanBPETokenizer.load(path)
    assert restored.encode(text) == tokenizer.encode(text)
    with pytest.raises(FileExistsError):
        tokenizer.save(path)
    legacy = KoreanBPETokenizer([(1, 2)])
    assert KoreanBPETokenizer.from_dict(legacy.to_dict()).to_dict() == legacy.to_dict()


def test_fast_rejects_empty_corpus():
    with pytest.raises(ValueError, match="empty"):
        FastBPETokenizer.train(iter([""]))
