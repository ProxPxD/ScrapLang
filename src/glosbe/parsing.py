from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from io import StringIO
from itertools import chain
from typing import Iterable, Any

import pandas as pd
import pydash as _
from bs4 import BeautifulSoup
from bs4.element import Tag
from decorator import decorator
from pandas import DataFrame
from pydash import chain as c
from requests import Response


class ParsingException(ValueError):
    pass


UNSET = object()


@decorator
def map_exceptions(func, *args, into: Any, raises: bool = True, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        if raises:
            raise into
        return into


# TODO: Rename from "Parser" to HtmlParser?
class Parser(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def _ensure_tag(cls, to_parse: Response | Tag | str) -> Tag:
        match to_parse:
            case Tag(): return to_parse
            case str(): return BeautifulSoup(to_parse, features="html.parser")
            case Response(): return cls._ensure_tag(to_parse.text)
            case _: raise ValueError(f'Cannot handle type {type(to_parse)} of {to_parse}!')

    @classmethod
    def ensure_tag(cls, func):
        @wraps(func)
        def wrapper(self, tag):
            return func(self, cls._ensure_tag(tag))
        return wrapper

    @classmethod
    @abstractmethod
    def parse(cls, to_parse: Response | Tag | str) -> Iterable:
        raise NotImplementedError


class TranslationKind(Enum):
    MAIN: str = 'main'
    LESS_FREQUENT: str = 'less-frequent'
    INDIRECT: str = 'indirect'


@dataclass(frozen=True)
class ParsedTranslation:
    kind: TranslationKind
    word: str
    gender: str = None
    pos: str = None

    @property
    def formatted(self) -> str:
        return self.word + (f' ({self.pos})' if self.pos else '') + (f' [{self.gender}]' if self.gender else '')
    # Thing of generality if more than two span case arises


class TranslationParser(Parser):
    @classmethod
    @Parser.ensure_tag
    def parse(cls, tag: Tag) -> Iterable[ParsedTranslation] | ParsingException:
        match mains := cls._parse_main_translations(tag):
            case Exception(): return mains
            case _: pass
        match less_freqs := cls.parse_less_frequent_translations(tag):
            case Exception(): less_freqs = []
            case _: pass
        return chain(mains, less_freqs)

    @classmethod
    def _parse_main_translations(cls, tag: Tag) -> Iterable[ParsedTranslation] | ParsingException:
        logging.debug('Parsing main translations')
        if not (trans_divs := tag.find_all('div', {'class': 'inline leading-10'})):
            return ParsingException('No translation div!')
        translations = []
        kinds = TranslationKind
        for trans_div in trans_divs:
            word = cls._get_translated_word(trans_div)
            spans = cls._get_spans(trans_div)
            # Assume for now
            pos = _.get(spans, '0.text')  # spans.map(c().get('0.text'))     # None is not Failure
            gender = _.get(spans, '1.text')  # None is not Failure
            translations.append(ParsedTranslation(kinds.MAIN, word, gender, pos))
        return translations

    @classmethod
    @Parser.ensure_tag
    def parse_less_frequent_translations(cls, tag: Tag) -> Iterable[ParsedTranslation | ParsingException]:
        logging.debug('Parsing less frequent translations')
        less_freq_tag = tag.find('ul', {'id': 'less-frequent-translations-container-0'})
        if not less_freq_tag:
            return ParsingException('No less frequent translations!')
        kinds = TranslationKind
        less_freqs = []
        for less_freq in less_freq_tag.find_all('li', {'class': 'break-inside-avoid-column'}):
            less_freqs.append(ParsedTranslation(kinds.LESS_FREQUENT, less_freq.text.replace('\n', '')))
        return less_freqs

    @classmethod
    @Parser.ensure_tag
    def parse_indirect_translations(cls, tag: Tag) -> Iterable[ParsedTranslation]:
        logging.debug('Parsing indirect translations')
        translation_buttons = tag.find_all('button', {'class': 'font-medium break-all flex-inline focus:outline-none'})
        kinds = TranslationKind
        indirects = []
        for button in translation_buttons:
            translation = button.find('span', {'class': 'text-primary-700 break-words font-medium text-base cursor-pointer'})
            indirects.append(ParsedTranslation(kinds.INDIRECT, translation.text.replace('\n', '')))
        return indirects

        # https://glosbe.com/uk/en/%D0%B7%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D0%B8%D1%81%D1%8F/fragment/indirect

    # To the below functions decorator or exception handling needed
    @classmethod
    def _get_translated_word(cls, translation_tag: Tag) -> str | ParsingException:
        return translation_tag.select_one('h3').text.replace('\n', '')

    @classmethod
    def _get_spans(cls, trans_tag: Tag) -> Tag | ParsingException:
        main_span = trans_tag.select_one('span', {'class': 'text-xxs text-gray-500'})
        return main_span.find_all('span')


class InflectionParser(Parser):
    @classmethod
    @Parser.ensure_tag
    def parse(cls, tag: Tag) -> Iterable[DataFrame] | ParsingException:
        logging.debug('Parsing inflection table')
        if not (table_tags := tag.select('div #grammar_0_0 table')):
            return ParsingException('No inflection table!')
        tables = [table for table_tag in table_tags for table in pd.read_html(StringIO(str(table_tag)), keep_default_na=False, header=None)]
        return tables[0]


@dataclass(frozen=True)
class ParsedDefinition:
    text: str
    examples: list[str]


@dataclass(frozen=True)
class ParsedDefinition:
    text: str
    examples: list[str]


class DefinitionParser(Parser):
    pos_form = re.compile('(?P<pos>.+)\n\n(?P<def>.+)', re.DOTALL)
    clean = c().join('').trim().trim(';')
    to_text = lambda tag: tag.text

    @classmethod
    @Parser.ensure_tag
    def parse(cls, tag: Tag) -> ParsingException | Iterable[ParsedDefinition]:
        logging.debug('Parsing definitions')
        if not (definition_tags := tag.find_all('li', {'class': 'pb-2'})):
            return ParsingException('No Definition Tags!')
        return map(cls._parse_definition, definition_tags)

    @classmethod
    def _parse_definition(cls, def_tag: Tag) -> ParsedDefinition:
        """
        Expects: The definition part up to a py-2 class that is a batch of examples
        """
        return ParsedDefinition(
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