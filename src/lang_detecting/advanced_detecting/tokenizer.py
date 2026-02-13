from abc import ABC, abstractmethod
from typing import Callable, Sequence

from toolz import valmap

from src.lang_detecting.advanced_detecting.model_io_mging import Class, \
    KindToVocab

CondMap = tuple[Callable, Callable]
KindToSpecs = dict[str, Sequence[CondMap]]

class Tokens:  # Req lowercase
    BOS = '<bos>'  # Boundary of Sequence


class ITokenizer(ABC):
    unknown = '<?>'
    @abstractmethod
    def tokenize(self, word: str | Sequence[str], *args):
        ...

    def __call__(self, word: str | Sequence[str], *args):
        return self.tokenize(word, *args)

    def __getitem__(self, word: str | Sequence[str], *args):
        return self.tokenize(word, *args)


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

    def tokenize(self, tokens: list[str]) -> Sequence[int]:
        return tuple([self.token2id.get(t, 0) if self.allow_unrecognized else self.token2id[t] for t in tokens])

    def detokenize(self, ids: list[int]) -> Sequence[str]:
        return tuple([self.id2token.get(i, self.unknown) if self.allow_unrecognized else self.id2token[i] for i in ids])


class GroupTokenizer(ITokenizer):
    def __init__(self, group: str):
        self.group = group

    def tokenize(self, word: str) -> Sequence[int]:
        return tuple([int(c in self.group) for c in word])


class MultiKindTokenizer(ITokenizer):
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            targets: list[Class] = None,
            allow_unrecognized: bool = True,
            kind_to_specs: KindToSpecs = None,
        ):
        self.kind2id = {k: i for i, k in enumerate(kinds_to_vocabs.keys())}
        self.id2kind: dict[int, str] = dict(map(reversed, self.kind2id.items()))
        self.kind_tokenizers: dict[str, Tokenizer] = valmap(lambda v: Tokenizer(v, allow_unrecognized=allow_unrecognized), kinds_to_vocabs)
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

    def detokenize_kind(self, id: int) -> str:
        return self.id2kind.get(id, self.unknown)

    def tokenize_target(self, target: str) -> Sequence[int]:
        return self.target_tokenizer([target])

    def detokenize_target(self, target: int) -> Sequence[str]:
        return self.target_tokenizer.detokenize([target])

    def tokenize_spec_groups(self, word: str | Sequence[str], kind: str) -> Sequence[Sequence[int]]:
        return tuple([tuple([int(cond(c)) for (cond, func) in self.kind_to_spec[kind]]) for c in word])

    def tokenize(self, word: str | Sequence[str], kind: str, targets: str | Sequence[str]):
        return self.tokenize_input(word, kind), self.tokenize_kind(kind), self.tokenize_target(targets), self.tokenize_spec_groups(word, kind)
