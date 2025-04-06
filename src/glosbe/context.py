from dataclasses import dataclass
from itertools import product, repeat
from typing import ClassVar, Iterable

import pydash as _
from pydash import chain as c


@dataclass(frozen=True, init=False)
class Context:
    words: list[str]
    from_lang: str
    to_langs: list[str]
    debug: bool
    groupby: str
    inflection: bool
    definition: bool

    to_filter: ClassVar[list[str]] = ['assume', 'args']

    def __init__(self, *confs: dict):
        own = c({}).merge_with(*confs, iteratee=_.curry(lambda a, b: a or b)).omit(self.to_filter).value()
        for key, val in own.items():
            object.__setattr__(self, key, val)

    @property
    def dest_pairs(self) -> Iterable[tuple[str, str]]:
        match self.groupby:
            case 'lang': return product(self.to_langs, self.words)
            case 'word': return ((to_lang, word) for word, to_lang in product(self.words, self.to_langs))
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def source_pairs(self) -> Iterable[tuple[[str, str]]]:
        return zip(repeat(self.from_lang), self.words)

    @property
    def url_triples(self) -> Iterable[tuple[str, str, str]]:
        for to_lang, word in self.dest_pairs:
            yield self.from_lang, to_lang, word


@dataclass(frozen=True)
class SubContext:
    from_lang: str
    to_lang: str
    word: str

    @property
    def lang(self) -> str:
        return self.from_lang or self.to_lang