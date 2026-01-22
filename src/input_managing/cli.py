from __future__ import annotations

import logging
import re
from argparse import ArgumentParser, Namespace, SUPPRESS, Action
from itertools import permutations, product, combinations, chain
from typing import Iterable

import pydash as _
from box import Box
from more_itertools import circular_shifts
from more_itertools import flatten
from ordered_set import OrderedSet
from pydash import chain as c

from src.conf import indirect, gather_data, infervia, groupby
from src.context import Context
from src.context_domain import UNSET, assume, at
from src.logutils import setup_logging


class AtSpecifierAction(Action):
    sides = 'ft'
    modes = 'oigd'

    @classmethod
    def mode_permutations(cls) -> Iterable[str]:
        for i in range(1, len(cls.modes) + 1):
            for comb in combinations(cls.modes, i):
                for perm in permutations(comb, len(comb)):
                    yield ''.join(perm)

    @classmethod
    def side_mode_fusions(cls) -> Iterable[str]:
        conflicting_like = {'to',}  #{f'{s}{m}' for m in cls.modes for s in cls.sides}
        perms = chain(
            cls.mode_permutations(),
            flatten((map(''.join, circular_shifts(side+perm)) for side, perm in product(cls.sides, cls.mode_permutations())))
        )
        filtereds = c(perms).reject(conflicting_like.__contains__).reject(lambda p: len(p) == 1).value()
        sorteds = sorted(filtereds, key=lambda p: (
            len(modes := (ps := OrderedSet(p)) & OrderedSet(cls.modes)),
            (' ' + p).index(next(iter(ps & OrderedSet(cls.sides)), ' ')),
            c(list(modes)).map(cls.modes.index).map(c().power(2)).sum().value(),
        ))
        return [f'-{group}' for group in sorteds]

    def __call__(self, parser, namespace, values, option_string=None):
        options = list(option_string.replace('-', ''))
        side = next((side for side in self.sides if side in options), 'none')
        try:
            options.remove(side)
        except ValueError:
            pass  # none

        setattr(namespace, 'at', side)
        for o in options:
            match o:
                case 'o': dest = 'wiktio'
                case 'i': dest = 'inflection'
                case 'g': dest = 'grammar'
                case 'd': dest = 'definition'
                case _: raise ValueError(f'Unexpected option: {o}')
            setattr(namespace, dest, True)


class CLI:
    def __init__(self, context: Context):
        self.context = context
        self._direct_arg = re.compile(r"^-[A-Za-z]\d")

    @property
    def parser(self) -> ArgumentParser:
        parser = ArgumentParser(
            prog='ScrapLang',
            description='Language Scrapping Program',
            epilog=''
        )
        parser = _.flow(self._add_base_args, self._add_execution_mode_args, self._add_setting_mode_args, self._add_loop_control_args)(parser)

        return parser

    def _add_base_args(self, parser: ArgumentParser) -> ArgumentParser:
        base_group = parser.add_argument_group(title='Base Arguments')
        base_group.add_argument('args', nargs='*', help='Words to translate, language to translate from and languages to translate to')
        base_group.add_argument('--words', '-words', '-words', '-word', '-w', nargs='+', default=[], help='Words to translate')
        base_group.add_argument('--from-langs', '--from-lang', '--from', '-from', '-f', dest='from_langs', nargs='+', default=[], help=(lang_help := 'Languages supported by glosbe.com'))
        base_group.add_argument('--to-langs', '--to-lang', '--to', '-to', '-t', dest='to_langs', nargs='+', default=[], help=lang_help)
        base_group.add_argument('--langs', '--lang', '-langs', '-lang', '-l', dest='langs', nargs='+', default=[], help=lang_help)
        return parser

    def _add_execution_mode_args(self, parser: ArgumentParser) -> ArgumentParser:
        # Translation Modes
        translation_mode_group = parser.add_argument_group(title='Translation Modes')

        translation_mode_group.add_argument('--at', '-at', help='Specify the from/to lang side to apply the mode to', choices=at, default='none')
        translation_mode_group.add_argument(*tuple(AtSpecifierAction.side_mode_fusions()), help='Side mode fusion', action=AtSpecifierAction, nargs=0, dest='_')
        translation_mode_group.add_argument('--wiktio', '-wiktio', '--overview', '-overview', '-o', action='store_true', default=False, help='Show morphological and etymological data')
        translation_mode_group.add_argument('--inflection', '--infl', '-infl', '-i', action='store_true', default=False, help='Show inflection')
        translation_mode_group.add_argument('--grammar', '-grammar', '-g', action='store_true', default=False, help='Show grammar info')
        translation_mode_group.add_argument('--definition', '--definitions', '-definition', '-definitions', '--def', '-def', '-d', action='store_true', default=False, help='Show word definitions')
        translation_mode_group.add_argument('--indirect', choices=indirect, default=UNSET, help='Turn on indirect translation')
        # CLI Reasoning Modes
        cli_reasoning_group = parser.add_argument_group(title='CLI Reasoning Modes')
        cli_reasoning_group.add_argument('--reverse', '--reversed', '-r', action='store_true', help='Reverse the from_lang(s) with the first to_lang')
        cli_reasoning_group.add_argument('--assume', choices=assume, default=UNSET, help='What to assume for a positional args in doubt of')
        cli_reasoning_group.add_argument('--gather-data', choices=gather_data, default=UNSET, help='What to gather user input for')
        cli_reasoning_group.add_argument('--infervia', '--iv', '-iv', choices=infervia, default=UNSET, help='How to infer the lang(s)')
        cli_reasoning_group.add_argument('--retrain', '--train', action='store_true', default=UNSET, help='Retrain the AI first')
        # Display Modes
        display_group = parser.add_argument_group(title='Display Modes')
        display_group.add_argument('--groupby', '-by', choices=groupby, default=UNSET, help='What to group the result translations by')
        # Developer Modes (groupless)
        parser.add_argument('--debug', action='store_true', help=SUPPRESS)
        parser.add_argument('--test', action='store_true', help=SUPPRESS)
        return parser

    # TODO: add and think through the settings displayal mode
    def _add_setting_mode_args(self, parser: ArgumentParser) -> ArgumentParser:
        setting_group = parser.add_argument_group(title='Setting')
        setting_group.add_argument('--set', '-set', '-s', action='append', nargs='+', default=[])
        setting_group.add_argument('--add', '-add', '-a', action='append', nargs='+', default=[])
        setting_group.add_argument('--delete', '-delete', '--del', '-del', action='append', nargs='+', default=[])
        return parser

    def _add_loop_control_args(self, parser: ArgumentParser) -> ArgumentParser:
        loop_control_group = parser.add_argument_group(title='Loop Control')
        loop_control_exclusive = loop_control_group.add_mutually_exclusive_group()
        loop_control_exclusive.add_argument('--loop', '-loop', action='store_true', default=UNSET, help='Enter a translation loop')
        loop_control_exclusive.add_argument('--exit', '-exit', action='store_false', default=UNSET, dest='loop', help='Exit loop')
        return parser

    def parse(self, args: list[str]) -> Namespace:
        if not args:
            self.parser.print_help()
            exit(0)  # change

        parsed, remaining = self.parser.parse_known_args(args)
        parsed.args += _.reject(remaining, '--'.__eq__)  # make test for this fix: t ksiądz -i pl
        setup_logging(parsed)
        self.context.update(**{**vars(parsed), 'words': UNSET, 'from_langs': UNSET, 'to_langs': UNSET}); logging.debug('Updating context in CLI')
        parsed = self._distribute_args(parsed)
        logging.debug(f'base Parsed: {parsed}')
        return parsed

    def _distribute_args(self, parsed: Namespace) -> Namespace:
        if parsed.langs and not parsed.from_langs:  # TODO: test both the following ifs
            parsed.from_langs = [parsed.langs.pop(0)]
        if parsed.langs:
            parsed.to_langs.extend(parsed.langs)

        parsed.orig_from_langs = parsed.from_langs
        parsed.orig_to_langs = parsed.to_langs
        match self.context.assume:
            case 'lang': return self._distribute_args_by_langs(parsed)
            case 'word':
                logging.debug(f'Assuming {parsed.args} are words!')
                parsed.words = parsed.args + parsed.words
                parsed.args = []
                return parsed
            case 'no' if parsed.args: raise ValueError(f'Could not resolve arguments: {parsed.args}')
            case 'no': return parsed
            case _: raise ValueError(f'Unexpected assume value: {self.context.assume}')

    def _distribute_args_by_langs(self, parsed: Namespace) -> Namespace:
        pot = Box(_.group_by(parsed.args, lambda arg: ('word', 'lang')[arg in self.context.langs]), default_box=True, default_box_attr=[])
        parsed.args = []

        # if not parsed.words and (assumed_word := self._assume_first_word(pot)):
        #     parsed.words.append(assumed_word)
        logging.debug(f'Potential langs: {pot.lang}')
        if pot.lang and not parsed.from_langs:
            from_langs = pot.lang.pop(0); logging.debug(f'Assuming "{from_langs}" should be in from_langs')
            parsed.from_langs = [from_langs]
        if pot.lang:
            parsed.to_langs = pot.lang + parsed.to_langs; logging.debug(f'Assuming "{pot.lang}" should be in to_langs')
            pot.lang = []
        if pot.word:
            parsed.words = pot.word + parsed.words; logging.debug(f'Assuming "{pot.word}" should be in words')
        return parsed
