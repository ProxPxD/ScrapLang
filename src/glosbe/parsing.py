from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from io import StringIO
from typing import Iterable, Callable, Any

import pandas as pd
import pydash as _
from bs4 import BeautifulSoup, NavigableString
from bs4.element import Tag
from decorator import decorator
from pandas import DataFrame
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
    # Thing of generality if more than two span case arises


class TranslationParser(Parser):
    @classmethod
    @Parser.ensure_tag
    def parse(cls, tag: Tag) -> Iterable[ParsedTranslation | ParsingException]:
        yield from cls._parse_main_translations(tag)
        match less_freqs := cls.parse_less_frequent_translations(tag):
            case Exception(): pass
            case _: yield from less_freqs

    @classmethod
    def _parse_main_translations(cls, tag: Tag) -> Iterable[ParsedTranslation] | ParsingException:
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
        # <ul class="columns-2 md:columns-4 text-primary-700 break-words font-medium text-base cursor-pointer py-2 pl-6" lang="de">
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
        if not (table_tags := tag.select('div #grammar_0_0 table')):
            return ParsingException('No inflection table!')
        return [table for table_tag in table_tags for table in pd.read_html(StringIO(str(table_tag)), keep_default_na=False, header=None)]


@dataclass(frozen=True)
class ParsedDefinition:
    text: str
    example: str


class DefinitionParser_(Parser):
    @classmethod
    @Parser.ensure_tag
    def parse(cls, tag: Tag) -> ParsingException | Iterable[ParsedDefinition]:
        if not (definition_tags := tag.find_all('li', {'class': 'pb-2'})):
            return ParsingException('No inflection table!')
        return map(cls._parse_definition, definition_tags)

    @classmethod
    def _parse_definition(cls, definition_tag: Tag) -> ParsedDefinition:
        definition_text = cls._parse_definition_text(definition_tag)
        example = cls._parse_example(definition_tag)
        return ParsedDefinition(definition_text, example)

    @classmethod
    def _parse_definition_text(cls, definition_tag: Tag) -> str:
        core_content = ''.join((content for content in definition_tag.contents if isinstance(content, (NavigableString, str))))
        return core_content\
            .removeprefix('\n')\
            .removesuffix('\n')\
            .replace('\n\n', ' ')\
            .replace('\n', ' ')\
            .removeprefix(' ')\
            .removeprefix(' ')

    @classmethod
    def _parse_example(cls, definition_tag: Tag) -> str:
        # TODO: think of railing?
        example_tag = definition_tag.select_one('div', {'class': 'border-l-2 pl-2 border-gray-200 text-gray-600 '})
        example = example_tag.text.replace('\n', '') if example_tag else ''
        if any((to_skip == example for to_skip in ('adjective', 'verb', 'noun'))):
            return ''
        return example
