from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Callable

import pydash as _
from pydash import chain as c
from toolz import valmap

if TYPE_CHECKING:
    from src.lang_detecting.advanced_detecting.model_io_mging import KindToTargets, KindToVocab

CondMap = tuple[Callable, Callable]
KindToSpecs = dict[str, Sequence[CondMap]]


@dataclass(frozen=True)
class Tokens:  # Req lowercase
    BOS: str = '<bos>'  # Boundary of Sequence
    PAD: str = '<pad>'  # Padding
    UNK: str = '<?>'    # Unknown

ENHANCE_TOKENS = list(asdict(Tokens()).values())


class ITokenizer(ABC):
    @abstractmethod
    def tokenize(self, word: str | Sequence[str]) -> list[int]:
        ...

    def __call__(self, word: str | Sequence[str]) -> list[int]:
        return self.tokenize(word)

    def __getitem__(self, word: str | Sequence[str]) -> list[int]:
        return self.tokenize(word)


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

    def tokenize(self, tokens: str | Sequence[str]) -> Sequence[int]:
        return tuple([self.token2id.get(t, self.unk_id) if self.is_allowed_unknown() else self.token2id[t] for t in tokens])

    def detokenize(self, ids: list[int]) -> Sequence[str] | str:
        return tuple([self.id2token.get(i, Tokens.UNK) for i in ids])


class GroupTokenizer(ITokenizer):
    def __init__(self, group: str):
        self.group = group

    def tokenize(self, word: str) -> Sequence[int]:
        return tuple([int(c in self.group) for c in word])


class MultiKindTokenizer:
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            kinds_to_targets: KindToTargets = None,
            kind_to_specs: KindToSpecs = None,
        ):
        self.kind2id = {k: i for i, k in enumerate(kinds_to_vocabs.keys())}
        self.id2kind: dict[int, str] = dict(map(reversed, self.kind2id.items()))
        self.kind_tokenizers: dict[str, Tokenizer] = valmap(Tokenizer, kinds_to_vocabs)
        self.kinds_to_targets = kinds_to_targets
        self.target_tokenizer = Tokenizer(c(kinds_to_targets.values()).flatten().sorted_uniq().value())
        self.kind_to_spec: KindToSpecs = kind_to_specs or {}

    @property
    def n_target_tokens(self) -> int:
        return self.target_tokenizer.n_tokens

    def get_tokenized_targets_for_kind(self, kind: str | int) -> list[int]:
        kind: str = self.detokenize_kind(kind) if isinstance(kind, int) else kind
        kind_targets: list[str] = self.kinds_to_targets[kind]
        return _.map_(kind_targets, self.tokenize_target)

    def tokenize_word(self, word: str | Sequence[str], kind: str) -> Sequence[int]:
        kind = self.detokenize_kind(kind) if isinstance(kind, int) else kind
        specs = self.kind_to_spec.get(kind, [])
        word = [(func(ch) if cond(ch) else ch) for ch in word for (cond, func) in specs]
        return self.kind_tokenizers[kind](word)

    def detokenize_word(self, ids: list[int], kind: str | int) -> Sequence[str]:
        kind = self.detokenize_kind(kind) if isinstance(kind, int) else kind
        return self.kind_tokenizers[kind].detokenize(ids)

    def tokenize_kind(self, kind: str) -> int:
        return self.kind2id[kind]

    def tokenize_common(self, val: str) -> int:
        return self.tokenize_word([val], self.detokenize_kind(0))[0]

    def detokenize_kind(self, kind_id: int) -> str:
        return self.id2kind.get(kind_id, Tokens.UNK)

    def tokenize_target(self, target: str) -> int:
        return self.target_tokenizer([target])[0]

    def tokenize_target_plural(self, target: str) -> Sequence[int]:
        return self.target_tokenizer([target])

    def detokenize_target(self, target: int) -> str:
        return self.target_tokenizer.detokenize([target])[0]

    def detokenize_targets(self, targets: list[int]) -> tuple[str, ...]:
        return tuple(_.map_(targets, self.detokenize_target))

    def detokenize_targets_as_onehot(self, targets: list[int]) -> tuple[str, ...]:
        targets = [i for i, is_target in enumerate(targets) if is_target]
        return self.detokenize_targets(targets)

    def detokenize_common(self, val: int) -> str:
        return self.detokenize_word([val], self.detokenize_kind(0))[0]

    def tokenize_spec_groups(self, word: str | Sequence[str], kind: str) -> Sequence[Sequence[int]]:
        return tuple([tuple([2*int(cond(c)) - 1 for (cond, func) in self.kind_to_spec[kind]]) for c in word])
