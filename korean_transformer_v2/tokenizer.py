"""A deterministic byte-level BPE tokenizer for Korean and arbitrary UTF-8."""

from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path


class KoreanBPETokenizer:
    EOS = 256
    FIRST_MERGE = 257

    def __init__(self, merges: Iterable[tuple[int, int]] = ()):
        self.merges = [tuple(pair) for pair in merges]
        self._bytes = {i: bytes([i]) for i in range(256)}
        self._ranks = {}
        for index, pair in enumerate(self.merges):
            if len(pair) != 2 or pair in self._ranks or any(part not in self._bytes for part in pair):
                raise ValueError("invalid BPE merge sequence")
            token_id = self.FIRST_MERGE + index
            self._ranks[pair] = token_id
            self._bytes[token_id] = self._bytes[pair[0]] + self._bytes[pair[1]]

    @property
    def vocab_size(self) -> int:
        return self.FIRST_MERGE + len(self.merges)

    @property
    def eos_token_id(self) -> int:
        return self.EOS

    @staticmethod
    def _merge(tokens, pair, token_id):
        result = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                result.append(token_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    @classmethod
    def train(cls, documents: Iterable[str], vocab_size: int = 512, min_frequency: int = 2):
        if vocab_size < cls.FIRST_MERGE or min_frequency < 1:
            raise ValueError("vocab_size must be at least 257 and min_frequency positive")
        sequences = []
        for document in documents:
            if not isinstance(document, str):
                raise TypeError("documents must contain strings")
            sequences.append(list(document.encode("utf-8")))
        if not any(sequences):
            raise ValueError("cannot train a tokenizer on empty documents")
        merges = []
        while cls.FIRST_MERGE + len(merges) < vocab_size:
            counts = Counter(pair for tokens in sequences for pair in zip(tokens, tokens[1:]))
            if not counts:
                break
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_frequency:
                break
            token_id = cls.FIRST_MERGE + len(merges)
            merges.append(pair)
            sequences = [cls._merge(tokens, pair, token_id) for tokens in sequences]
        return cls(merges)

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            candidates = [(self._ranks[pair], pair) for pair in zip(tokens, tokens[1:]) if pair in self._ranks]
            if not candidates:
                break
            token_id, pair = min(candidates)
            tokens = self._merge(tokens, pair, token_id)
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Iterable[int], *, skip_eos: bool = True) -> str:
        pieces = []
        for token in tokens:
            token = int(token)
            if token == self.EOS and skip_eos:
                continue
            if token not in self._bytes:
                raise ValueError(f"unknown token ID: {token}")
            pieces.append(self._bytes[token])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "korean_byte_bpe_v1", "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, state: dict):
        if state.get("type") == "hf_byte_bpe_v1":
            return FastBPETokenizer.from_dict(state)
        if state.get("type") != "korean_byte_bpe_v1" or not isinstance(state.get("merges"), list):
            raise ValueError("invalid tokenizer state")
        return cls(state["merges"])

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class FastBPETokenizer(KoreanBPETokenizer):
    """Rust byte-level BPE. Token IDs are private to each trained tokenizer."""

    def __init__(self, backend):
        self.backend = backend
        if self.eos_token_id is None:
            raise ValueError("tokenizer is missing EOS")

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size()

    @property
    def eos_token_id(self):
        return self.backend.token_to_id("<|endoftext|>")

    @classmethod
    def train(cls, documents, vocab_size=8000, min_frequency=2):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        if vocab_size < 257 or min_frequency < 1:
            raise ValueError("vocab_size must be at least 257 and min_frequency positive")
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      special_tokens=["<|endoftext|>"], show_progress=False)
        count = 0
        def checked():
            nonlocal count
            for document in documents:
                if not isinstance(document, str):
                    raise TypeError("documents must contain strings")
                if document:
                    count += 1
                    yield document
        backend.train_from_iterator(checked(), trainer=trainer)
        if not count:
            raise ValueError("cannot train a tokenizer on empty documents")
        return cls(backend)

    def encode(self, text, *, add_eos=False):
        ids = self.backend.encode(text, add_special_tokens=False).ids
        return ids + [self.eos_token_id] if add_eos else ids

    def decode(self, tokens, *, skip_eos=True):
        return self.backend.decode([int(token) for token in tokens], skip_special_tokens=skip_eos)

    def to_dict(self):
        return {"type": "hf_byte_bpe_v1", "json": self.backend.to_str()}

    @classmethod
    def from_dict(cls, state):
        from tokenizers import Tokenizer
        if state.get("type") != "hf_byte_bpe_v1":
            raise ValueError("invalid tokenizer state")
        return cls(Tokenizer.from_str(state["json"]))
