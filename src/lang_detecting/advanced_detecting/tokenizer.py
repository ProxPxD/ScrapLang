import pydash as _
from pydash import map_values
from toolz import valmap


class SingleTokenizer:
    def __init__(self, vocab: str):
        self.vocab_size = len(vocab)
        self.char2id = {c: i for i, c in enumerate(vocab)}

    def tokenize(self, word: str) -> list[int]:
        return [self.char2id.get(c, self.vocab_size) for c in word]

class MultiTokenizer:
    def __init__(self, kinds_to_vocabs: dict[str, str]):
        self.subtokenizers: dict[str, SingleTokenizer] = valmap(SingleTokenizer, kinds_to_vocabs)
        self.kind2id = {k: i for i, k in enumerate(kinds_to_vocabs)}

    def tokenize_word(self, word: str, kind: str) -> list[int]:
        return self.subtokenizers[kind].tokenize(word)

    def tokenize_kind(self, kind: str) -> int:
        return self.kind2id[kind]
