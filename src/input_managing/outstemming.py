from __future__ import annotations

import logging
import re
from functools import cached_property
from itertools import combinations, chain
from typing import Sequence, Iterable, Optional, Callable

import pydash as _
import regex
import toolz.curried.operator as op
from pydash import chain as c
from toolz import valfilter


class ReSymbolSet(frozenset[str]):
    def __new__(cls, elems: str | Iterable[str]) -> ReSymbolSet:
        return super().__new__(cls, elems)

    def get_any(self) -> str:
        return next(iter(self))

    @cached_property
    def together(self) -> str:
        or_last = lambda e: int(e == '|')
        return "".join(map(re.escape, sorted(self, key=or_last)))

    @cached_property
    def group(self) -> str:
        return f'(?:{self.together})'

    @cached_property
    def star(self) -> str:
        return f'{self.group}*'

    @cached_property
    def plus(self) -> str:
        return f'{self.group}+'

    @cached_property
    def any(self) -> str:
        return f'[{self.together}]'

    @cached_property
    def not_(self) -> str:
        return f'[^{self.together}]'

    def __or__(self, other):
        return ReSymbolSet(set(self)|set(other))


class Outstemmer:
    def __init__(self,
            left_brackets: str | Sequence[str] = '[({',
            right_brackets: str | Sequence[str] = '])}',
            alt_seps: str | Sequence[str] = ',|',
            postcutters: str | Sequence[str] = '/',
            precutters: str | Sequence[str] = '\\',
            enders: str | Sequence[str] = '.',
        ):
        self._left_brackets: ReSymbolSet = ReSymbolSet(left_brackets)
        self._right_brackets: ReSymbolSet = ReSymbolSet(right_brackets)
        self._alt_seps: ReSymbolSet = ReSymbolSet(alt_seps)
        self._postcutters: ReSymbolSet = ReSymbolSet(postcutters)
        self._precutters: ReSymbolSet = ReSymbolSet(precutters)
        self._enders: ReSymbolSet = ReSymbolSet(enders)
        for (n1, s1), (n2, s2) in combinations(self._symbol_groups.items(), 2):
            if common := s1 & s2:
                raise ValueError(f'Parameters {n1[1:]} and {n2[1:]} should have no common symbol: {common}')

        # r'(?>/+)(?!\d)'
        # self._cutters_sequence = re.compile(f'({self._postcutters.any_of})+\D')
        lb, rb, s, c, e = self._left_brackets, self._right_brackets, self._alt_seps, self._postcutters, self._enders
        lbt, rbt = lb.together, rb.together
        self._bracketed = re.compile(fr'[{lbt}][^{lbt}{rbt}]+[{rbt}]')
        ca = c.any
        ea = e.any
        sa = s.any
        secn = ReSymbolSet(s|e|c).not_
        self._invalid_seq = re.compile(fr'{ca}{2,}\d')
        self._cutter_seq = re.compile(fr'({ca}+)(?!\d)')
        self._is_cutted = re.compile(fr'{ca}')
        cut_scope = fr'{ca}(?P<n>\d+)(?:({secn}+){sa})*({secn}+)?'  # f'{ca}+(?:({secn}*){sa})*({secn}*){ea}?'
        self._cutted = regex.compile(cut_scope)

    @property
    def _symbol_groups(self) -> dict[str, set[str]]:
        return valfilter(c().is_set(), vars(self))

    def outstem(self, word: str) -> list:
        logging.debug(f'outstemming "{word}"')
        modes = 'bracketed', 'cutted'
        for mode in modes:
            stemmer = getattr(self, f'_outstem_{mode}')
            if stemmeds := stemmer(word):
                return stemmeds
        else:
            return [word]

    def flatmap_outstem(self, words: Iterable[str], *others: str) -> list[str]:
        return c(chain(words, others)).map(self.outstem).flatten().map(c().trim()).filter().uniq().value()

    def _outstem_bracketed(self, word: str) -> Optional[list[str]]:
        if not (matched := self._bracketed.search(word)):
            return None
        logging.debug(f'matched bracketed "{matched}"')

        pattern = matched.group(0)
        brackets = slice(*matched.span(0))
        alts = re.split(self._alt_seps.any, pattern[1:-1])
        if len(alts) == 1:
            alts.insert(0, '')
        outstemmeds = [word[:brackets.start] + alt + word[brackets.stop:] for alt in alts]
        return self.flatmap_outstem(outstemmeds)

    @classmethod
    def count(cls, string: str, chars: Iterable[str]) -> int:
        return sum(string.count(ch) for ch in chars)

    def join_outstem_syntax(self, words: list[str]) -> list[str]:
        bracket_diff = [self.count(word, self._left_brackets) - self.count(word, self._right_brackets) for word in words]
        if _.every(bracket_diff, c().eq(0)):
            return words
        if sum(bracket_diff) != 0:
            raise ValueError('Incorrect bracketing')
        joined_words, buffer, gauge = [], [], 0
        for part, diff in zip(words, bracket_diff):
            buffer.append(part)
            gauge += diff
            if not gauge:
                joined_words.append(' '.join(buffer))
                buffer = []
        return joined_words

    def _outstem_cutted(self, word: str) -> Optional[list[str]]:
        if not self._is_cutted.search(word):
            return None
        if self._invalid_seq.search(word):
            raise ValueError(f'Cannot have more than one cutter "{self._postcutters.together}" with a number')
        def repl(m: re.Match):
            c = self._postcutters.get_any()
            return f'{c}{len(m.group(1))}'

        wordy = self._cutter_seq.sub(repl, word)
        matched = self._cutted.search(wordy)
        n = int(matched.group('n'))
        start = matched.start(0)
        outer_slice = slice(start-n, start)
        orig = wordy[:outer_slice.stop].rstrip('_')
        cut = wordy[:outer_slice.start]
        to_puts = c(matched.captures(2, 3)).flatten().map(c().trim_start('_')).filter().value() or ['']
        putteds = c(to_puts).map(op.add(cut)).value()
        rest = wordy[matched.end(0):]
        bare_rest = rest.lstrip('.')
        fulls = c(putteds).map(c().add(rest.lstrip('.'))).value()
        if rest.startswith('.'):
            orig += bare_rest
        return self.flatmap_outstem([orig.rstrip('_')], *fulls)
        # rest = wordy[matched.end(0):]
        # rest_match = self.after_ender(rest)
        # immediate, continuation = rest_match.groups()
        # # if immediate and not continuation:
        # if immediate:
        #     puts = c(puts).map(c().add(immediate)).value()
        # joined = c((orig, *puts)).map(c().add(continuation or '')).value()
        # return self.flatmap_outstem(joined)
