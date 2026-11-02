"""A deterministic character-level tokenizer with one unknown-token ID."""

from collections.abc import Iterable


class CharTokenizer:
    def __init__(self, characters: Iterable[str]) -> None:
        self.itos = list(characters)
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("characters must be unique")
        self.stoi = {character: index for index, character in enumerate(self.itos)}
        self.unk_token_id = len(self.itos)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos) + 1  # Include unknown token.

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(character, self.unk_token_id) for character in text]

    def decode(self, token_ids) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        result = []
        for token_id in token_ids:
            if isinstance(token_id, list):
                result.append(self.decode(token_id))
            elif 0 <= int(token_id) < len(self.itos):
                result.append(self.itos[int(token_id)])
            else:
                result.append("�")
        return "".join(result)

    def to_dict(self) -> dict:
        return {"characters": self.itos, "unknown_token_id": self.unk_token_id}

    @classmethod
    def from_dict(cls, values: dict) -> "CharTokenizer":
        tokenizer = cls(values["characters"])
        if values.get("unknown_token_id", tokenizer.unk_token_id) != tokenizer.unk_token_id:
            raise ValueError("checkpoint tokenizer has an incompatible unknown-token ID")
        return tokenizer
