import logging
import re
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from itertools import product
from tkinter.scrolledtext import example
from typing import Optional

import pandas as pd
import pydash as _
from bs4.element import ResultSet, Tag
from more_itertools import bucket
from numpy.lib.recfunctions import join_by
from pandas import DataFrame
from pydash import chain as c, flow, partial

from ..core.parsing import Parser, ParsingException, Result, with_ensured_tag


class TransResultKind(Enum):
    MAIN = 'main'
    INDIRECT = 'indirect'
    LESS_FREQS = 'less_freqs'

@dataclass(frozen=True)
class TransResult(Result):
    kind: str
    word: str
    gender: str = None
    pos: str = None

    @property
    def formatted(self) -> str:
        return self.word + (f' ({self.pos})' if self.pos else '') + (f' [{self.gender}]' if self.gender else '')


class TranslationParser(Parser):
    @classmethod
    @with_ensured_tag
    def parse(cls, tag: Tag) -> list[TransResult] | ParsingException:
        if isinstance(mains := cls._parse_main_translations(tag), Exception):
            return mains
        if isinstance(less_freqs := cls.parse_less_frequent_translations(tag), Exception):
            less_freqs = []
        return _.concat(mains, less_freqs)

    @classmethod
    def _parse_main_translations(cls, tag: Tag) -> list[TransResult] | ParsingException:
        logging.debug('Parsing main translations')
        if not (main_section := tag.select_one('article div div section')) or not (trans_divs := main_section.find_all('div', {'class': 'inline leading-10'})):
            return ParsingException('No translation div!', tag)
        translations = []
        for trans_div in trans_divs:
            word = cls._get_translated_word(trans_div)
            spans = cls._get_spans(trans_div)
            # Assume for now
            pos = _.get(spans, '0.text')  # spans.map(c().get('0.text'))     # None is not Failure
            gender = _.get(spans, '1.text')  # None is not Failure
            translations.append(TransResult(TransResultKind.MAIN, word, gender, pos))
        return translations

    @classmethod
    @with_ensured_tag
    def parse_less_frequent_translations(cls, tag: Tag) -> list[TransResult] | ParsingException:
        logging.debug('Parsing less frequent translations')
        less_freq_tag = tag.find('ul', {'id': 'less-frequent-translations-container-0'})
        if not less_freq_tag:
            return ParsingException('No less frequent translations!')
        less_freqs = []
        for less_freq in less_freq_tag.find_all('li', {'class': 'break-inside-avoid-column'}):
            less_freqs.append(TransResult(TransResultKind.LESS_FREQS, less_freq.text.replace('\n', '')))
        return less_freqs

    @classmethod
    @with_ensured_tag
    def parse_indirect_translations(cls, tag: Tag) -> list[TransResult] | ParsingException:
        logging.debug('Parsing indirect translations')
        if not (translation_buttons := tag.find_all('button', {'class': 'font-medium break-all flex-inline focus:outline-none'})):
            return ParsingException('No indirect translations')
        indirects = []
        for button in translation_buttons:
            translation = button.find('span', {'class': 'text-primary-700 break-words font-medium text-base cursor-pointer'})
            indirects.append(TransResult(TransResultKind.INDIRECT, translation.text.replace('\n', '')))
        return indirects

        # https://glosbe.com/uk/en/%D0%B7%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D0%B8%D1%81%D1%8F/fragment/indirect

    # To the below functions decorator or exception handling needed
    @classmethod
    def _get_translated_word(cls, translation_tag: Tag) -> str | ParsingException:
        return translation_tag.select_one('h3').text.replace('\n', '')

    @classmethod
    def _get_spans(cls, trans_tag: Tag) -> ResultSet[Tag] | ParsingException:
        main_span = trans_tag.select_one('span', {'class': 'text-xxs text-gray-500'})
        return main_span.find_all('span')


class InflectionParser(Parser):
    _bold_selector = ':not(table b)'

    @classmethod
    @with_ensured_tag
    def parse(cls, tag: Tag) -> DataFrame | str | ParsingException:
        logging.debug('Parsing inflection')
        if not isinstance(table := cls.parse_table(tag), Exception):
            return table
        logging.debug(table.args[0])
        if not isinstance(grammar := cls.parse_grammar(tag), Exception):
            return grammar
        return ParsingException('No inflection table nor grammar info!')

    @classmethod
    def parse_table(cls, tag: Tag) -> Optional[DataFrame]:
        if not (table_tags := tag.select('div #grammar_0_0 table')):
            return ParsingException('No inflection table!')
        to_table = flow(str, StringIO, partial(pd.read_html, keep_default_na=False, header=None))
        to_table_or_none = c().apply_catch(to_table, {ValueError}, None)
        table = c(table_tags).flat_map(to_table_or_none).reject(_.is_none).head().value()
        return table

    @classmethod
    @with_ensured_tag
    def parse_grammar(cls, tag: Tag) -> Optional[list[list[str]]] | ParsingException:
        if not (grammar_items := tag.select('div #grammar_0_0 ul li')):  # TODO: test: (en) man; (sv) mus, sida
            return ParsingException('No grammar info!')
        examples = c(grammar_items).filter(cls.is_grammar_item_valid).map(cls.extract_grammar_examples).filter(cls.discard_examples).value()
        uniques = cls.uniq_grammar_example_batches(examples)
        return uniques

    @classmethod
    def is_grammar_item_valid(cls, item: Tag) -> bool:
        bolds = cls.get_bolds(item)
        return bolds or not item.select('table') and ',' in item.text

    @classmethod
    def get_bolds(cls, tag: Tag) -> list[Tag]:
        return [b for b in tag.find_all('b') if not b.find_parent('table')]

    @classmethod
    def extract_grammar_examples(cls, item: Tag) -> list[str]:
        examples = _.map_(cls.get_bolds(item), c().get('text')) or item.text.split(',')
        return c(examples).map(c().trim()).value()

    @classmethod
    def discard_examples(cls, examples: list[str]) -> bool:
        text = ''.join(examples)
        return '\xa0' not in text and 'Note:' not in text and len(examples) > 1

    @classmethod
    def uniq_grammar_example_batches(cls, example_batch: list[list[str]]) -> list[list[str]]:
        set_to_examples = {}
        for examples in example_batch:
            set_to_examples.setdefault(frozenset(examples), examples)

        for a, b in product(list(set_to_examples.keys()), repeat=2):
            if a < b and a in set_to_examples:
                del set_to_examples[a]

        return list(set_to_examples.values())

@dataclass(frozen=True)
class DefResult(Result):
    text: str
    examples: list[str]


class DefinitionParser(Parser):
    pos_form = re.compile('(?P<pos>.+)\n\n(?P<def>.+)', re.DOTALL)
    clean = c().join('').trim().trim(';')
    to_text = lambda tag: tag.text

    @classmethod
    @with_ensured_tag
    def parse(cls, tag: Tag) -> list[DefResult] | ParsingException:
        logging.debug('Parsing definitions')
        if not (definition_tags := tag.find_all('li', {'class': 'pb-2'})):
            return ParsingException('No Definition Tags!')
        return _.map_(definition_tags, cls._parse_definition)

    @classmethod
    def _parse_definition(cls, def_tag: Tag) -> DefResult:
        """
        Expects: The definition part up to a py-2 class that is a batch of examples
        """
        return DefResult(
            text=cls._parse_text(def_tag),
            examples=cls._parse_examples(def_tag)
        )

    @classmethod
    def _parse_text(cls, def_tag: Tag) -> str:
        text = c(def_tag.contents)\
            .take_while(c().get('attrs.class',  []).filter_(c().is_equal('py-2')).is_empty())\
            .map_(cls.to_text)\
            .apply(cls.clean)\
            .value()
        if not (matched := cls.pos_form.search(text)):  # No PoS
            return text
        pos = matched.group('pos').replace('\n', ' ')
        return f"({pos}) {matched.group('def')}"

    @classmethod
    def _parse_examples(cls, def_tag: Tag) -> list:
        if not (batch := def_tag.find('div', class_='py-2')):
            return []
        examples = c(batch)\
            .apply(lambda tag: tag.find_all('div'))\
            .map_(cls.to_text)\
            .map_(cls.clean)\
            .filter_()\
            .value()
        return examples
