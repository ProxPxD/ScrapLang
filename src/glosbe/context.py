from dataclasses import dataclass
from functools import cache
from itertools import product, repeat
from typing import ClassVar, Iterable, Optional

import pydash as _
from pydash import chain as c


@dataclass(frozen=True, init=False)
class Context:
    words: tuple[str]
    from_lang: str
    to_langs: tuple[str]

    inflection: bool
    definition: bool

    debug: bool
    groupby: str
    indirect: bool

    _to_filter: ClassVar[tuple[str]] = ('assume', 'args')

    def __init__(self, *confs: dict):
        own = c({}).merge_with(*confs, iteratee=_.curry(lambda a, b: a or b)).omit(self._to_filter).value()
        for key, val in own.items():
            if isinstance(val, list):
                val = tuple(val)
            object.__setattr__(self, key, val)

    @property
    def all_langs(self) -> list:
        return [self.from_lang, *self.to_langs]

    @property
    def dest_pairs(self) -> Iterable[tuple[Optional[str], str]]:
        to_langs = self.to_langs or [None]
        match self.groupby:
            case 'lang': return product(to_langs, self.words)
            case 'word': return ((to_lang, word) for word, to_lang in product(self.words, to_langs))
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def source_pairs(self) -> Iterable[tuple[[str, str]]]:
        return zip(repeat(self.from_lang), self.words)

    @property
    def url_triples(self) -> Iterable[tuple[str, Optional[str], str]]:
        for to_lang, word in self.dest_pairs:
            yield self.from_lang, to_lang, word

    @property
    @cache
    def n_members(self) -> int:
        match self.groupby:
            case 'lang': return len(self.words)
            case 'word': return len(self.to_langs)
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def grouped_url_triples(self) -> Iterable:
        for i, (from_lang, to_lang, word) in enumerate(self.url_triples):
            is_first = self._is_first(i)
            is_last = self._is_last(i)
            yield is_first, is_last, (from_lang, to_lang, word)

    def _is_first(self, i) -> bool:
        return self.n_members == 0 or i % self.n_members == 0

    def _is_last(self, i) -> bool:
        return self.n_members == 0 or i % self.n_members == self.n_members - 1

    @property
    def prefix_type(self) -> str:
        match self.groupby:
            case 'lang': return 'word'
            case 'word': return 'to_lang'
            case _: raise ValueError(f'Unexpected groupby value: {self.groupby}')


@dataclass(frozen=True)
class SubContext:
    from_lang: str
    to_lang: str
    word: str

    @property
    def lang(self) -> str:
        return self.from_lang or self.to_lang