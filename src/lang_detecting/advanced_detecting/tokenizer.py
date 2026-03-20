from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Sequence

from toolz import valmap
import pydash as _

from src.lang_detecting.advanced_detecting.model_io_mging import Class, \
    KindToVocab

CondMap = tuple[Callable, Callable]
KindToSpecs = dict[str, Sequence[CondMap]]


@dataclass
class Tokens:  # Req lowercase
    BOS = '<bos>'  # Boundary of Sequence
    PAD = '<pad>'  # Padding
    UNK = '<?>'    # Unknown


class ITokenizer(ABC):
    @abstractmethod
    def tokenize(self, word: str | Sequence[str], *args):
        ...

    def __call__(self, word: str | Sequence[str], *args):
        return self.tokenize(word, *args)

    def __getitem__(self, word: str | Sequence[str], *args):
        return self.tokenize(word, *args)


class Tokenizer(ITokenizer):
    def __init__(self, tokens: list[str] | str):
        tokens = list(tokens or [])
        self.token2id = {k: i for i, k in enumerate(tokens)}
        self.id2token: dict[int, str] = dict(map(reversed, self.token2id.items()))

    def is_allowed_unknown(self) -> bool:
        return Tokens.UNK in self.token2id

    @cached_property
    def unk_id(self) -> int:
        return self.token2id[Tokens.UNK]

    @property
    def n_tokens(self) -> int:
        return len(self.token2id)

    def tokenize(self, tokens: list[str]) -> Sequence[int]:
        return tuple([self.token2id.get(t, self.unk_id) if self.is_allowed_unknown() else self.token2id[t] for t in tokens])

    def detokenize(self, ids: list[int]) -> Sequence[str] | str:
        return tuple([self.id2token.get(i, Tokens.UNK) for i in ids])


class GroupTokenizer(ITokenizer):
    def __init__(self, group: str):
        self.group = group

    def tokenize(self, word: str) -> Sequence[int]:
        return tuple([int(c in self.group) for c in word])


class MultiKindTokenizer(ITokenizer):
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            targets: list[Class] = None,
            kind_to_specs: KindToSpecs = None,
        ):
        self.kind2id = {k: i for i, k in enumerate(kinds_to_vocabs.keys())}
        self.id2kind: dict[int, str] = dict(map(reversed, self.kind2id.items()))
        self.kind_tokenizers: dict[str, Tokenizer] = valmap(Tokenizer, kinds_to_vocabs)
        self.target_tokenizer = Tokenizer(targets)
        self.kind_to_spec: KindToSpecs = kind_to_specs or {}

    @property
    def n_target_tokens(self) -> int:
        return self.target_tokenizer.n_tokens

    def tokenize_input(self, input: str | Sequence[str], kind: str) -> Sequence[int]:
        specs = self.kind_to_spec.get(kind, [])
        input = [(func(ch) if cond(ch) else ch) for ch in input for (cond, func) in specs]
        return self.kind_tokenizers[kind](input)

    def detokenize_input(self, ids: list[int], kind: str) -> Sequence[str]:
        return self.kind_tokenizers[kind].detokenize(ids)

    def tokenize_kind(self, kind: str) -> int:
        return self.kind2id[kind]

    def tokenize_common(self, val: str) -> int:
        return self.tokenize_input([val], self.detokenize_kind(0))[0]

    def detokenize_kind(self, id: int) -> str:
        return self.id2kind.get(id, Tokens.UNK)

    def tokenize_target(self, target: str) -> Sequence[int]:
        return self.target_tokenizer([target])

    def detokenize_target(self, target: int) -> str:
        return self.target_tokenizer.detokenize([target])[0]

    def detokenize_targets(self, targets: list[int]) -> tuple[str, ...]:
        return tuple(_.map_(targets, self.detokenize_target))

    def detokenize_targets_as_onehot(self, targets: list[int]) -> tuple[str, ...]:
        targets = [i for i, is_target in enumerate(targets) if is_target]
        return self.detokenize_targets(targets)

    def detokenize_common(self, val: int) -> str:
        return self.detokenize_input([val], self.detokenize_kind(0))[0]

    def tokenize_spec_groups(self, word: str | Sequence[str], kind: str) -> Sequence[Sequence[int]]:
        return tuple([tuple([2*int(cond(c)) - 1 for (cond, func) in self.kind_to_spec[kind]]) for c in word])

    def tokenize(self, word: str | Sequence[str], kind: str, targets: str | Sequence[str]):
        return self.tokenize_input(word, kind), self.tokenize_kind(kind), self.tokenize_target(targets), self.tokenize_spec_groups(word, kind)
