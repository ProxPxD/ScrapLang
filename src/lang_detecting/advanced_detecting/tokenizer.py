from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import cache
from typing import Callable, Collection, Sequence

import pydash as _
from pydash import map_values
from pydash import chain as c
from toolz import valmap

from src.lang_detecting.advanced_detecting.model_io_mging import KindToSpecialGroup, ALL, SpecialGroup, Class, \
    KindToVocab, Vocab


class ITokenizer(ABC):
    @abstractmethod
    def tokenize(self, word: str | list[str]):
        ...

    def __call__(self, word: str | list[str]):
        return self.tokenize(word)

    def __getitem__(self, word: str | list[str]):
        return self.tokenize(word)


class Tokenizer(ITokenizer):
    def __init__(self, tokens: list[str] | str, allow_unrecognized: bool = False):
        tokens = list(tokens or [])
        self.allow_unrecognized: bool = allow_unrecognized
        self.token2id = {k: i for i, k in enumerate(tokens, start=int(self.allow_unrecognized))}
        self.id2token: dict[int, str] = dict(map(reversed, self.token2id.items()))

    @property
    def n_tokens(self) -> int:
        return self.n_known_tokens + int(self.allow_unrecognized)

    @property
    def n_known_tokens(self) -> int:
        return len(self.token2id)

    def tokenize(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, 0) if self.allow_unrecognized else self.token2id[t] for t in tokens]

    def detokenize(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(i, '<?>') if self.allow_unrecognized else self.id2token[i] for i in ids]


class GroupTokenizer(ITokenizer):
    def __init__(self, group: str):
        self.group = group

    def tokenize(self, word: str) -> list[int]:
        return [int(c in self.group) for c in word]


class MultiKindTokenizer:
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            targets: list[Class] = None,
            allow_unrecognized: bool = True,
            kind_to_specs: dict[str, Sequence[Callable]] = None,
        ):
        self.kind2id = {k: i for i, k in enumerate(kinds_to_vocabs.keys())}
        self.id2kind: dict[int, str] = dict(map(reversed, self.kind2id.items()))
        self.kind_tokenizers: dict[str, Tokenizer] = valmap(lambda v: Tokenizer(v, allow_unrecognized=allow_unrecognized), kinds_to_vocabs)
        self.target_tokenizer = Tokenizer(targets)
        self.kind_to_spec: dict[str, Sequence[Callable]] = kind_to_specs or {}

    @property
    def n_target_tokens(self) -> int:
        return self.target_tokenizer.n_tokens

    def tokenize_input(self, input: str, kind: str) -> list[int]:
        return self.kind_tokenizers[kind](input)

    def detokenize_input(self, ids: list[int], kind: str) -> list[str]:
        return self.kind_tokenizers[kind].detokenize(ids)

    def tokenize_kind(self, kind: str) -> int:
        return self.kind2id[kind]

    def detokenize_kind(self, id: int) -> str:
        return self.id2kind.get(id, '<?>')

    def tokenize_target(self, target: str) -> list[int]:
        return self.target_tokenizer([target])

    def detokenize_target(self, target: int) -> list[str]:
        return self.target_tokenizer.detokenize([target])

    def tokenize_spec_groups(self, word: str | list[str], kind: str) -> list[list[int]]:
        return [[int(spec(c)) for spec in self.kind_to_spec[kind]] for c in word]

    def tokenize(self, word: str | list[str], kind: str, targets: list[str]):
        return self.tokenize_input(word, kind), self.tokenize_kind(kind), self.tokenize_target(targets), self.tokenize_spec_groups(word, kind)
