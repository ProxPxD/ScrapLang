from __future__ import  annotations

import logging
import re
from functools import cached_property, cache
from itertools import takewhile, product, combinations
from typing import Sequence, Iterable, Callable, Optional

import pydash as _
from more_itertools import padded
from pydash import chain as c
import logging
import re

import pydash as _
from pydash import chain as c
from toolz import valfilter

class ReSymbolSet(frozenset):
    def __new__(cls, elems: str | Sequence[str]) -> ReSymbolSet:
        return super().__new__(cls, elems)

    @cache
    def alt(self) -> str:
        return f'(?:{("|".join(map(re.escape, self)))})'

    @cache
    def zero_or_more(self) -> str:
        return f'{self.alt()}*'

    @cache
    def one_or_more(self) -> str:
        return f'{self.alt()}+'


class Outstemmer:
    def __init__(self,
            left_brackets: str | Sequence[str] = '[({',
            right_brackets: str | Sequence[str] = '])}',
            alt_seps: str | Sequence[str] = ',|',
            postcutters: str | Sequence[str] = '/',
            precutters: str | Sequence[str] = '\\',
        ):
        self._left_brackets = ReSymbolSet(left_brackets)
        self._right_brackets = ReSymbolSet(right_brackets)
        self._alt_seps = ReSymbolSet(alt_seps)
        self._postcutters = ReSymbolSet(postcutters)
        self._precutters = ReSymbolSet(precutters)
        for (n1, s1), (n2, s2) in combinations(self._symbol_groups.items(), 2):
            if common := s1 & s2:
                raise ValueError(f'Parameters {n1[1:]} and {n2[1:]} should have no common symbol: {common}')

    @property
    def _symbol_groups(self) -> dict[str, set[str]]:
        return valfilter(c().is_set(), vars(self))

    @cached_property
    def bracketed(self):
        lbs, rbs = self._left_brackets.alt(), self._right_brackets.alt()
        return re.compile(fr'[{lbs}][^{lbs}{rbs}]+[{rbs}]').search

    @cached_property
    def postcutted(self):
        return re.compile(self._postcutters.one_or_more()).search

    @cached_property
    def precutted(self):
        return re.compile(self._precutters.one_or_more()).search

    def outstem(self, word: str) -> list:
        # TODO: anhi test (and improve for "normal[ize[d]]")
        # TODO: test: rett[ig[het]] to generate three words
        # TODO: [lønn^{s:wtf}opp^gjør]: [lønn, opp, gjør, lønnsoppgjør]  # TODO: test and how to handle both "|" and "^" together? Prohibit?
        # TODO: [teil^nehmen|haben]  # I think: yeah, forbid
        # TODO: Why not just [teil][nehmen]?
        # TODO: test for trimming '[password] [manager]'
        # Cause I want to take them literally out and not in the compound. 3 together would break, but maybe other operator would be better
        logging.debug(f'outstemming "{word}"')
        modes = 'bracketed', 'postcutted', 'precutted'  # TODO: anhi: make precutted
        for mode in modes:
            stemmer = getattr(self, f'_outstem_{mode}')
            if stemmeds := stemmer(word):
                return stemmeds
        else:
            return [word]

    def _flatmap_outstem(self, words: Iterable[str]) -> list[str]:
        return c(words).map(self.outstem).flatten().map(c().trim()).filter().uniq().value()

    def _outstem_bracketed(self, word: str) -> Optional[list[str]]:
        if not (matched := self.bracketed(word)):
            return None
        logging.debug(f'matched bracketed "{matched}"')

        pattern = matched.group(0)
        alts = re.split(self._alt_seps.alt(), pattern[1:-1])
        if len(alts) == 1:
            alts.insert(0, '')
        outstemmeds = [word.replace(pattern, alt) for alt in alts]
        return self._flatmap_outstem(outstemmeds)

    def _outstem_postcutted(self, word: str) -> Optional[list[str]]:
        if not (matched := self.postcutted(word)):
            return None
        logging.debug(f'matched postcutted "{matched}"')

        start = matched.start(0) - len(matched.group(0))
        end = matched.end(0)
        bare = word[:start]
        orig = bare + word[start]
        novel = bare + word[end:]
        return self._flatmap_outstem([orig, novel])

    def _outstem_precutted(self, word: str) -> Optional[list[str]]:
        if not (matched := self.precutted(word)):
            return None
        logging.debug(f'matched precutted "{matched}"')

    @classmethod
    def count(cls, string: str, chars: Iterable[str]) -> int:
        return sum(string.count(ch) for ch in chars)

    def join_outstem_syntax(self, words: list[str]) -> list[str]:
        bracket_diff = [self.count(word, self._left_brackets) - self.count(word, self._right_brackets) for word in words]
        if _.every(bracket_diff, c().eq(0)):
            return words
        joined_words, buffer, gauge = [], [], 0
        for part, diff in zip(words, bracket_diff):
            buffer.append(part)
            gauge += diff
            if not gauge:
                joined_words.append(' '.join(buffer))
                buffer = []
        return joined_words

