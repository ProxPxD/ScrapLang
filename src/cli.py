import json
import logging
import re
import sys
from argparse import ArgumentParser, Namespace, SUPPRESS
from functools import reduce
from typing import Optional
from xml.dom.expatbuilder import parseFragmentString

import pydash as _
from box import Box
from pydash import chain as c

from .context import Context
from .logutils import setup_logging
from .resouce_managing.data_gathering import DataGatherer


class Outstemming:
    """
    P - Parenthesis
    L/R - Left/Right
    E - Escaped
    """

    LP = '[('
    RP = '])'
    P = LP + RP
    ELP = f'\\{LP}'
    ERP = f'\\{RP}'
    EP = ELP + ERP

    parenthesised = re.compile(fr'[{ELP}][^{ELP}{ERP}]+[{ERP}]').search
    slashed = re.compile('/+').search
    baskslashed = re.compile('\\+').search

    @classmethod
    def outstem(cls, complex_word: str) -> list:
        # TODO: anhi test (and improve for "normal[ize[d]]")
        flat_outstem = c().map(cls.outstem).flatten()
        logging.debug(f'outstemming "{complex_word}"')
        if matched := cls.parenthesised(complex_word):
            logging.debug(f'matched "{matched}"')
            pattern = matched.group(0)
            alts = re.split('[/|]', pattern[1:-1])
            if len(alts) == 1:
                alts = ['', alts[0]]
            outstemmeds = [complex_word.replace(pattern, alt) for alt in alts]
            return flat_outstem(outstemmeds)
        if matched := cls.slashed(complex_word):
            logging.debug(f'matched "{matched}"')
            start = matched.start(0) - len(matched.group(0))
            end = matched.end(0)
            bare = complex_word[:start]
            orig = bare + complex_word[start]
            novel = bare + complex_word[end:]
            return flat_outstem([orig, novel])
        # TODO: anhi: make baskslashes
        return [complex_word]

    @classmethod
    def count(cls, string: str, chars: str) -> int:
        return sum(string.count(ch) for ch in chars)

    @classmethod
    def join_outstem_syntax(cls, words: list[str]) -> list[str]:
        parenthesis_diff = [cls.count(word, cls.LP) - cls.count(word, cls.RP) for word in words]
        if _.every(parenthesis_diff, c().eq(0)):
            return words
        joined_words, buffer, gauge = [], [], 0
        for part, diff in zip(words, parenthesis_diff):
            buffer.append(part)
            gauge += diff
            if not gauge:
                joined_words.append(' '.join(buffer))
                buffer = []
        return joined_words


class CLI:
    def __init__(self, conf: Box, context: Context, data_gatherer: DataGatherer = None):
        self.conf: Box = conf
        self.context = context  # TODO: convert to using context and data_gatherer instead of conf
        self.data_gatherer = data_gatherer or DataGatherer(context)

    @property
    def parser(self) -> ArgumentParser:
        parser = ArgumentParser(
            prog='GlosbeTranslator',
            description='Translation Program',
            epilog=''
        )
        parser = _.flow(self._add_base_args, self._add_execution_mode_args, self._add_setting_mode_args, self._add_loop_control_args)(parser)

        return parser

    def _add_base_args(self, parser: ArgumentParser) -> ArgumentParser:
        base_group = parser.add_argument_group(title='Base Arguments')
        base_group.add_argument('args', nargs='*', help='Words to translate, language to translate from and languages to translate to')
        base_group.add_argument('--words', '-words', '-w', nargs='+', default=[], help='Words to translate')
        base_group.add_argument('--from-lang', '--from', '-from', '-f', dest='from_lang', help=(lang_help := 'Languages supported by glosbe.com'))
        base_group.add_argument('--to-lang', '--to', '-to', '-t', '-l', dest='to_langs', nargs='+', default=[], help=lang_help)
        # TODO: think of adding a flag --lang/-l being generic for both to- and from- langs
        return parser

    def _add_execution_mode_args(self, parser: ArgumentParser) -> ArgumentParser:
        # Translation Modes
        translation_mode_group = parser.add_argument_group(title='Translation Modes')
        translation_mode_group.add_argument('--inflection', '--infl', '-infl', '-i', '--conjugation', '--conj', '-conj', '-c', '--declension', '--decl', '-decl', '--table', '-tab', action='store_true', default=False, help='#todo')
        translation_mode_group.add_argument('--definition', '--definitions', '--def', '-def', '-d', action='store_true', default=False, help='#todo')
        translation_mode_group.add_argument('--indirect', choices=['on', 'off', 'fail', 'conf'], help='Turn on indirect translation')
        # CLI Reasoning Modes
        cli_reasoning_group = parser.add_argument_group(title='CLI Reasoning Modes')
        cli_reasoning_group.add_argument('--reverse', '--reversed', '-r', action='store_true', help='Reverse the from_lang with the first to_lang')
        cli_reasoning_group.add_argument('--assume', choices=['lang', 'word', 'no'], help='What to assume for a positional args in doubt of')
        cli_reasoning_group.add_argument('--gather-data', '--gd', '-gd', choices=['all', 'ai', 'time', 'off', 'conf'], help='What to gather user input for')
        cli_reasoning_group.add_argument('--infervia', '--iv', '-iv', choices=['all', 'ai', 'time', 'last', 'off', 'conf'], help='How to infer the lang(s)')
        # Display Modes
        display_group = parser.add_argument_group(title='Display Modes')
        display_group.add_argument('--groupby', '-by', choices=['lang', 'word'], help='What to group the result translations by')
        # Developer Modes (groupless)
        parser.add_argument('--debug', action='store_true', help=SUPPRESS)
        parser.add_argument('--test', action='store_true', help=SUPPRESS)
        return parser

    # TODO: anhi: add and think through the displayal mode
    def _add_setting_mode_args(self, parser: ArgumentParser) -> ArgumentParser:
        setting_group = parser.add_argument_group(title='Setting')
        setting_group.add_argument('--set', '-set', '-s', action='append', nargs='+', default=[])
        setting_group.add_argument('--add', '-add', '-a', action='append', nargs='+', default=[])
        setting_group.add_argument('--delete', '-delete', '--del', '-del', action='append', nargs='+', default=[])
        return parser

    def _add_loop_control_args(self, parser: ArgumentParser) -> ArgumentParser:
        loop_control_group = parser.add_argument_group(title='Loop Control')
        loop_control_exclusive = loop_control_group.add_mutually_exclusive_group()
        loop_control_exclusive.add_argument('--loop', action='store_true', default=None, help='Enter a translation loop')
        loop_control_exclusive.add_argument('--exit', action='store_false', default=None, dest='loop', help='Exit loop')
        return parser

    def parse(self, args: list[str] | str = None) -> Namespace:
        parsed = self.parse_base(args or sys.argv[1:])
        self.data_gatherer.gather_short_mem(parsed)
        parsed = self.process_parsed(parsed)
        return parsed

    def parse_base(self, args: list[str]):
        if len(args) == 0:
            self.parser.print_help()
            exit(0)  # change

        args = [a for arg in args for a in arg.split('\xa0')]
        parsed, remaining = self.parser.parse_known_args(args)
        parsed.args += remaining  # make test for this fix: t ksiądz -i pl
        setup_logging(parsed)
        parsed = self._distribute_args(parsed)
        logging.debug(f'base Parsed: {parsed}')
        return parsed

    def process_parsed(self, parsed: Namespace) -> Namespace:
        parsed = self._word_outstemming(parsed)
        parsed = self._fill_default_args(parsed)
        parsed = self._reverse_if_needed(parsed)
        origs = list(parsed.words)
        parsed = self._apply_mapping(parsed)
        parsed.mapped = [o != m for o, m in zip(origs, parsed.words)]
        logging.debug(f'Processed: {parsed}')
        return parsed

    def _distribute_args(self, parsed: Namespace) -> Namespace:
        assume = parsed.assume or self.conf.assume
        if assume == 'word':
            logging.debug(f'Assuming {parsed.args} are words!')
            parsed.words = parsed.args + parsed.words
            return parsed
        if assume == 'no' and parsed.args:
            raise ValueError(f'Could not resolve arguments: {parsed.args}')

        # assume == lang
        pot = Box(_.group_by(parsed.args, lambda arg: ('word', 'lang')[arg in self.conf.langs]), default_box=True, default_box_attr=[])
        parsed.args = []

        if not parsed.words and (assumed_word := self._assume_first_word(pot)):
            parsed.words.append(assumed_word)
        logging.debug(f'Potential langs: {pot.lang}')
        if pot.lang and not parsed.from_lang:
            from_lang = pot.lang.pop(0); logging.debug(f'Assuming "{from_lang}" should be in from_lang')
            parsed.from_lang = from_lang
        if pot.lang and not parsed.to_langs:
            parsed.to_langs = pot.lang + parsed.to_langs; logging.debug(f'Assuming "{pot.lang}" should be in to_langs')
            pot.lang = []
        if pot.word:
            parsed.words += pot.word; logging.debug(f'Assuming "{pot.word}" should be in words')
        return parsed

    def _assume_first_word(self, pot: Box) -> Optional[str]:
        if pot.word:
            word = pot.word.pop(0)
        elif len(pot.lang) > 2:
            word = pot.lang.pop(0)
        else:
            logging.debug('No word to assume!')
            return None
        logging.debug(f'Assuming "{word}" should be in words')
        return word

    def _word_outstemming(self, parsed: Namespace) -> Namespace:
        parsed.words = Outstemming.join_outstem_syntax(parsed.words)
        parsed.words = [outstemmed for word in parsed.words for outstemmed in Outstemming.outstem(word)]
        return parsed

    def _fill_default_args(self, parsed: Namespace) -> Namespace:
        parsed = self._predict_langs(parsed)
        parsed = self._fill_last_used(parsed)
        return parsed

    def _predict_langs(self, parsed: Namespace) -> Namespace:
        # TODO: How to handle Cyrillic written with latin and memory?
        return parsed

    def _fill_last_used(self, parsed: Namespace) -> Namespace:
        used = _.filter_([parsed.from_lang] + parsed.to_langs)
        pot_defaults = [lang for lang in self.conf.langs if lang not in used]; logging.debug(f'Potential defaults: {pot_defaults}')
        if len(self.conf.langs) < (n_needed := int(not parsed.from_lang) + int(not parsed.to_langs)):
            raise ValueError(f'Config has not enough defaults! Needed {n_needed}, but possible to choose only: {pot_defaults}')
        # Do not require to translate on definition or inflection
        if not parsed.to_langs and (parsed.definition or parsed.inflection):
            n_needed -= 1
        to_fill = pot_defaults[:n_needed]; logging.debug(f'Chosen defaults: {to_fill}')
        if not parsed.from_lang and to_fill:
            from_lang = to_fill.pop(0); logging.debug(f'Filling from_lang with "{from_lang}"')
            parsed.from_lang = from_lang
        if not parsed.to_langs and to_fill:
            to_lang = to_fill.pop(0); logging.debug(f'Filling to_lang with "{to_lang}"')
            parsed.to_langs.append(to_lang)
        return parsed

    def _reverse_if_needed(self, parsed: Namespace) -> Namespace:
        if parsed.reverse:
            old_from, old_first_to = parsed.from_lang, parsed.to_langs[0]
            logging.debug(f'Reversing: {old_from, old_first_to} => {old_first_to, old_from}')
            parsed.from_lang = old_first_to
            parsed.to_langs[0] = old_from
        return parsed

    def _apply_mapping(self, parsed: Namespace) -> Namespace:
        # todo: test żurawel (regex)
        lang_mapping: Box
        if not (lang_mapping := self.conf.mappings.get(parsed.from_lang)):
            return parsed
        ordered_lang_mapping = [lang_mapping] if isinstance(lang_mapping, dict) else lang_mapping
        logging.debug(f'Applying mapping for "{parsed.from_lang}" with map:\n{json.dumps(lang_mapping, indent=4, ensure_ascii=False)}')
        for lang_mapping in ordered_lang_mapping:
            patts, repls = zip(*sorted(lang_mapping.to_dict().items(), key=c().get(0).size(), reverse=True))
            lang_mapping: list = list(map(lambda patt, repl: (re.compile(patt), repl), patts, repls))
            logging.debug(f'from_lang_map: {lang_mapping}')
            parsed.words = [
                reduce(lambda w, patt_repl: patt_repl[0].sub(patt_repl[1], w), lang_mapping, word)
                for word in parsed.words
            ]
        return parsed