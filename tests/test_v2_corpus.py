import torch
from korean_transformer_v2.corpus import DiskCorpus
from korean_transformer_v2.tokenizer import KoreanBPETokenizer


def test_disk_corpus_deduplicates_and_maps_tokens(tmp_path):
    docs = [f"한국어 문서 {i}" for i in range(100)]
    corpus = DiskCorpus(iter(docs + docs), tmp_path)
    train, val = list(corpus.documents()), list(corpus.documents("validation"))
    assert len(train) + len(val) == 100
    assert not set(train) & set(val)
    tokenizer = KoreanBPETokenizer()
    tokens = corpus.encode(tokenizer, "train")
    assert tokens.device.type == "cpu"
    assert tokens.tolist() == [i for text in train for i in tokenizer.encode(text, add_eos=True)]
    fingerprint = corpus.fingerprint
    corpus.close()
    again = DiskCorpus(iter(docs), tmp_path)
    assert again.fingerprint == fingerprint
    again.close()
