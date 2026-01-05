from abc import ABC, abstractmethod

import pydash as _
from pydash import map_values
from pydash import chain as c
from toolz import valmap

from src.lang_detecting.advanced_detecting.model_io_mging import KindToGroupedVocab, ALL, GroupedVocab


class ITokenizer(ABC):
    @abstractmethod
    def tokenize(self, word: str):
        ...

    def __call__(self, word: str):
        return self.tokenize(word)


class Tokenizer(ITokenizer):
    def __init__(self, vocab: str):
        self.n_vocab = len(vocab)
        self.vocab2id = {k: i for i, k in enumerate(vocab)}

    def tokenize(self, word: str) -> list[int]:
        return [self.vocab2id.get(c, self.n_vocab) for c in word]


class GroupTokenizer(ITokenizer):
    def __init__(self, group: str):
        self.group = group

    def tokenize(self, word: str) -> list[int]:
        return [int(c in self.group) for c in word]

class SingleKindTokenizer(ITokenizer):
    def __init__(self, grouped_vocab: GroupedVocab):
        self.vocab_tokenizer = Tokenizer(grouped_vocab.pop(ALL))
        self.group_tokenizers: dict[str, GroupTokenizer] = valmap(GroupTokenizer, grouped_vocab)

    @property
    def tokenizers(self) -> list[ITokenizer]:
        return [self.vocab_tokenizer, *list(self.group_tokenizers.values())]

    def tokenize(self, word: str) -> list[list[int]]:
        return [[tokenizer(c) for tokenizer in self.tokenizers] for c in word]

class MultiKindTokenizer:
    def __init__(self, kinds_to_grouped_vocabs: KindToGroupedVocab):
        self.kind2id = {k: i for i, k in enumerate(kinds_to_grouped_vocabs.keys())}
        self.kind_tokenizers: dict[str, SingleKindTokenizer] = valmap(SingleKindTokenizer, kinds_to_grouped_vocabs)

    def tokenize_word(self, word: str, kind: str) -> list[list[int]]:
        return self.kind_tokenizers[kind](word)

    def tokenize_kind(self, kind: str) -> int:
        return self.kind2id[kind]

    def tokenize(self, word: str, kind: str):
        return self.tokenize_word(word, kind), self.tokenize_kind(kind)
