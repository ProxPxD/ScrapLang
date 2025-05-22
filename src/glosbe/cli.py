import logging
import sys
# from smartcli import Parameter, HiddenNode, Cli, Root, CliCollection, Flag
from argparse import ArgumentParser, Namespace
from functools import reduce
from typing import Optional

import pydash as _
from box import Box
from pydash import chain as c

from .constants import supported_languages


class CLI:
    def __init__(self, conf: Box):
        self.conf = conf
        logging.debug(f'Conf: {conf.to_yaml()}')

    @property
    def parser(self) -> ArgumentParser:
        parser = ArgumentParser(
            prog='GlosbeTranslator',
            description='Translation Program',
            epilog=''
        )
        supported_langs_msg = f'Supported languages: {" ".join(supported_languages.keys())}'

        # Main args
        parser.add_argument('args', nargs='*', help='Words to translate, language to translate from and languages to translate to')
        parser.add_argument('--words', '-w', nargs='+', default=[], help='Words to translate')
        parser.add_argument('--from', '--from-lang', '-f', dest='from_lang', help=supported_langs_msg)
        parser.add_argument('--to', '--to-lang', '-t', '-l', dest='to_langs', nargs='+', default=[], help=supported_langs_msg)
        # Loop Control
        loop_control = parser.add_mutually_exclusive_group()
        loop_control.add_argument('--loop', action='store_true', default=None, help='Enter a translation loop')
        loop_control.add_argument('--exit', action='store_false', default=None, dest='loop', help='Exit loop')
        # TODO: think of adding a flag --lang/-l being generic for both to- and from- langs
        parser.add_argument('--inflection', '--infl', '-infl', '-i', '--conjugation', '--conj', '-conj', '-c', '--declension', '--decl', '-decl', '--table', '-tab', action='store_true', default=False, help='#todo')
        parser.add_argument('--definition', '--definitions', '--def', '-def', '-d', action='store_true', default=False, help='#todo')
        parser.add_argument('--indirect', choices=['on', 'off', 'fail'], help='Turn on indirect translation')
        # Cli Conf
        parser.add_argument('--assume', choices=['lang', 'word', 'no'], help='What to assume for a positional args in doubt of')
        parser.add_argument('--reverse', '--reversed', '-r', action='store_true', help='Reverse the from_lang with the first to_lang')
        # Display Conf
        parser.add_argument('--groupby', '-by', choices=['lang', 'word'], help='What to group the result translations by')
        # Developer Conf
        parser.add_argument('--debug', action='store_true')
        return parser

    def parse(self, args=None) -> Namespace:
        parsed = self.parse_base(args)
        self.process_parsed(parsed)
        return parsed

    def parse_base(self, args: list[str]):
        if len(args := args or sys.argv) == 1:
            self.parser.print_help()
            exit(0)  # change

        args = [a for arg in args for a in arg.split('\xa0')][1:]
        parsed = self.parser.parse_args(args)
        parsed = self._distribute_args(parsed)
        logging.debug(f'base Parsed: {parsed}')
        return parsed

    def process_parsed(self, parsed: Namespace) -> Namespace:
        parsed = self._fill_default_args(parsed)
        parsed = self._reverse_if_needed(parsed)
        parsed = self._apply_mapping(parsed)
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
        pot = Box(_.group_by(parsed.args, lambda arg: ('word', 'lang')[arg in supported_languages]), default_box=True, default_box_attr=[])
        parsed.args = []

        if not parsed.words and (assumed_word := self._assume_first_word(pot)):
            parsed.words.append(assumed_word)
        logging.debug(f'Potential langs: {pot.lang}')
        if pot.lang and not parsed.from_lang:
            from_lang = pot.lang.pop(0)
            logging.debug(f'Assuming: "{from_lang}" is from_lang')
            parsed.from_lang = from_lang
        if pot.lang and not parsed.to_langs:
            singular = len(pot.lang) == 1
            logging.debug(f'Assuming: {pot.lang} {"is" if singular else "are"} to_lang{"" if singular else "s"}')
            parsed.to_langs = pot.lang + parsed.to_langs
            pot.lang = []
        if pot.word:
            singular = len(pot.word) == 1
            logging.debug(f'Assuming: {pot.lang} {"is" if singular else "are"} word{"" if singular else "s"}')
            parsed.words += pot.word
        return parsed

    def _assume_first_word(self, pot: Box) -> Optional[str]:
        if pot.word:
            word = pot.word.pop(0)
        elif len(pot.lang) > 2:
            word = pot.lang.pop(0)
        else:
            logging.debug('No word to assume!')
            return None
        logging.debug(f'Assuming: "{word}" is word')
        return word

    def _fill_default_args(self, parsed: Namespace) -> Namespace:
        parsed = self._predict_langs(parsed)
        parsed = self._fill_last_used(parsed)
        return parsed

    def _predict_langs(self, parsed: Namespace) -> Namespace:
        # How to handle Cyrillic written with latin and memory?
        return parsed

    def _fill_last_used(self, parsed: Namespace) -> Namespace:
        used = _.filter_([parsed.from_lang] + parsed.to_langs)
        pot_defaults = [lang for lang in self.conf.langs if lang not in used]
        logging.debug(f'Potential defaults: {pot_defaults}')
        if len(self.conf.langs) < (n_needed := int(not parsed.from_lang) + int(not parsed.to_langs)):
            raise ValueError(f'Config has not enough defaults! Needed {n_needed}, but possible to choose only: {pot_defaults}')
        # Do not require to translate on definition or inflection
        if not parsed.to_langs and (parsed.definition or parsed.inflection):
            n_needed -= 1
        to_fill = pot_defaults[:n_needed]
        logging.debug(f'Chosen defaults: {to_fill}')
        if not parsed.from_lang and to_fill:
            from_lang = to_fill.pop(0)
            logging.debug(f'Filling from_lang with {from_lang}')
            parsed.from_lang = from_lang
        if not parsed.to_langs and to_fill:
            to_lang = to_fill.pop(0)
            logging.debug(f'Filling to_lang with {to_lang}')
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
        if from_lang_map := self.conf.mappings.get(parsed.from_lang):
            logging.debug(f'Applying mapping for {parsed.from_lang}')
            from_lang_map = sorted(from_lang_map.items(), key=c().get(0).size(), reverse=True)
            parsed.words = [
                reduce(lambda w, orig_dest: w.replace(*orig_dest), from_lang_map, word)
                for word in parsed.words
            ]
        return parsed