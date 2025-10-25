import logging
import re
from functools import cached_property
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


class Outstemmer:
    def __init__(self,
            left_brackets: str | Sequence[str] = '[({',
            right_brackets: str | Sequence[str] = '])}',
            alt_seps: str | Sequence[str] = ',|',
            postcutters: str | Sequence[str] = '/',
            precutters: str | Sequence[str] = '\\',
        ):
        self._left_brackets = set(left_brackets)
        self._right_brackets = set(right_brackets)
        self._alt_seps = set(alt_seps)
        self._postcutters = set(postcutters)
        self._precutters = set(precutters)
        for (n1, s1), (n2, s2) in combinations(self._symbol_groups.items(), 2):
            if s1 ^ s2:
                raise ValueError(f'Parameters {n1[1:]} and {n2[1:]} should have no common symbol')

    @property
    def _symbol_groups(self) -> dict[str, set[str]]:
        return valfilter(c().is_set(), vars(self))

    @classmethod
    def _to_regex_group(cls, symbols: Iterable[str]) -> str:
        return f'(?:{re.escape("|".join(symbols))})'

    @cached_property
    def _left_brackets_regex(self) -> str:
        return self._to_regex_group(self._left_brackets)

    @cached_property
    def _right_brackets_regex(self) -> str:
        return self._to_regex_group(self._right_brackets)

    @cached_property
    def _alt_seps_regex(self) -> str:
        return f'[{self._to_regex_group(self._alt_seps)}]'

    @cached_property
    def _postcutters_regex(self) -> str:
        return f'{self._to_regex_group(self._postcutters)}+'

    @cached_property
    def _precutters_regex(self) -> str:
        return f'{self._to_regex_group(self._precutters)}+'

    @cached_property
    def bracketed(self):
        lbs, rbs = self._left_brackets_regex, self._right_brackets_regex
        return re.compile(fr'[{lbs}][^{lbs}{rbs}]+[{rbs}]').search

    @cached_property
    def postcutted(self):
        return re.compile(self._postcutters_regex).search

    @cached_property
    def precutted(self):
        return re.compile(self._precutters_regex).search

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
        alts = re.split(self._alt_seps_regex, pattern[1:-1])
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

