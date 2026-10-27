from data.tokenizer import CharTokenizer


def test_tokenizer_round_trip_and_unknown_character():
    tokenizer = CharTokenizer.from_text("hello world")
    text = "hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode("?")) == "�"
    assert tokenizer.vocab_size == len(set(text)) + 1
