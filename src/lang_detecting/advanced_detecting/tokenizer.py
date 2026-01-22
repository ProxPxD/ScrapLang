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


class SpecGroup:
    def __init__(self, kind_cond: Callable[[str], bool] | str | Collection[str], token_cond: Callable[[str], bool]):
        match kind_cond:
            case None: kind_cond = lambda _: True
            case str(): kind_cond = (kind_cond,).__contains__
            case _ if isinstance(kind_cond, Collection): kind_cond = kind_cond.__contains__
            case _: pass
        self.kind_cond = kind_cond
        self.token_cond = token_cond

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
        self.token2id = {k: i for i, k in enumerate(tokens)}

    @property
    def n_tokens(self) -> int:
        return len(self.token2id) + int(self.allow_unrecognized)

    def tokenize(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, self.n_tokens) if self.allow_unrecognized else self.token2id[t] for t in tokens]


class GroupTokenizer(ITokenizer):
    def __init__(self, group: str):
        self.group = group

    def tokenize(self, word: str) -> list[int]:
        return [int(c in self.group) for c in word]


class MultiKindTokenizer:
    def __init__(self,
                 kinds_to_vocabs: KindToVocab,
                 outputs: list[Class] = None,
                 allow_unrecognized: bool = True,
                 kind_to_specs: dict[str, Sequence[Callable]] = None,
                 ):
        self.kind2id = {k: i for i, k in enumerate(kinds_to_vocabs.keys())}
        self.kind_tokenizers: dict[str, Tokenizer] = valmap(lambda v: Tokenizer(v, allow_unrecognized=allow_unrecognized), kinds_to_vocabs)
        self.output_tokenizer = Tokenizer(outputs)
        self.kind_to_spec: dict[str, Sequence[Callable]] = kind_to_specs or {}

    @property
    def n_outputs(self) -> int:
        return self.output_tokenizer.n_tokens

    def tokenize_input(self, input: str, kind: str) -> list[int]:
        return self.kind_tokenizers[kind](input)

    def tokenize_kind(self, kind: str) -> int:
        return self.kind2id[kind]

    def tokenize_output(self, output: str | list[str]) -> list[int]:
        return self.output_tokenizer([output])

    def tokenize_spec_groups(self, word: str | list[str], kind: str) -> list[list[int]]:
        return [[int(spec(c)) for spec in self.kind_to_spec[kind]] for c in word]

    def tokenize(self, word: str | list[str], kind: str, outputs: list[str]):
        return self.tokenize_input(word, kind), self.tokenize_kind(kind), self.tokenize_output(outputs), self.tokenize_spec_groups(word, kind)
