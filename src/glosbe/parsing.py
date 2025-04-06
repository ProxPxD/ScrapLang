from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, ResultSet
from bs4.element import Tag
from decorator import decorator
from pydash import chain as c
from requests import Response
from returns.maybe import Maybe
from returns.result import safe, Result, Success, Failure


class ParsingException(ValueError):
    pass


UNSET = object()


@decorator
def railway(func, *args, on_failure=UNSET, on_exception=UNSET, on_none=UNSET, on_falsy=UNSET, is_failure=lambda x: x is None, **kwargs):
    result: Result = safe(func)(*args, **kwargs)
    if on_failure is not UNSET:
        result = result.alt(lambda _: Failure(on_failure))
        result = result.bind(lambda x: Success(x) if not is_failure(x) else Failure(on_failure))
    else:
        if on_exception is not UNSET:
            result = result.alt(lambda _: Failure(on_exception))
        if on_none is not UNSET:
            result = result.bind(lambda x: Success(x) if x is not None else Failure(on_none))
        elif on_falsy is not UNSET:
            result = result.bind(lambda x: Success(x) if not x else Failure(on_falsy))
    return result


# TODO: Rename from "Parser" to HtmlParser?
class Parser(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def ensure_tag(cls, to_parse: Response | Tag | str) -> Tag:
        match to_parse:
            case Tag(): return to_parse
            case str(): return BeautifulSoup(to_parse, features="html.parser")
            case Response(): return cls.ensure_tag(to_parse.text)
            case _: raise ValueError(f'Cannot handle type {type(to_parse)} of {to_parse}!')

    @classmethod
    @safe
    def parse(cls, to_parse: Response | Tag | str) -> Result[Iterable, ParsingException]:
        return cls._parse(cls.ensure_tag(to_parse))

    @classmethod
    @abstractmethod
    def _parse(cls, to_parse: Tag) -> Iterable:
        raise NotImplementedError


@dataclass(frozen=True)
class ParsedTranslation:
    word: str
    gender: str = None
    pos: str = None
    # Thing of generality if more than two span case arises


class TranslationParser_(Parser):
    @classmethod
    def _parse(cls, tag: Tag) -> Iterable[Result[tuple(str, str, str), ParsingException]]:
        if not (trans_divs := tag.find_all('div', {'class': 'inline leading-10'})):
            raise ParsingException('No translation div!')
        for trans_div in map(Success, trans_divs):
            word = trans_div.bind(cls._get_translated_word)
            spans: Result = trans_div.bind(cls._get_spans)
            # Assume for now
            pos = spans.map(c().get('0.text'))     # None is not Failure
            gender = spans.map(c().get('1.text'))  # None is not Failure
            yield Result.do(ParsedTranslation(w, g, p) for w in word for g in gender for p in pos)

    @classmethod
    @railway(on_failure=ParsingException('No word!'))
    def _get_translated_word(cls, translation_tag: Tag) -> str:
        return translation_tag.select_one('h3').text.replace('\n', '')

    @classmethod
    @railway(on_failure=ParsingException('No spans!'))
    def _get_spans(cls, trans_tag: Tag) -> ResultSet[Tag]:
        main_span = trans_tag.select_one('span', {'class': 'text-xxs text-gray-500'})
        return main_span.find_all('span')


class ConjugationParser(Parser):
    @classmethod
    def _parse(cls, tag: Tag):
        if not (tables := tag.select('div #grammar_0_0 table')):
            raise ParsingException('No inflection table!')
        for table in tables:
            yield pd.read_html(StringIO(str(table)), keep_default_na=False, header=None)


@dataclass(frozen=True)
class ParsedDefinition:
    text: str
    example: str


class DefinitionParser_(Parser):
    @classmethod
    def _parse(cls, tag: Tag):
        if not (definition_tags := tag.find_all('li', {'class': 'pb-2'})):
            raise ParsingException('No inflection table!')
        definitions = map(cls._parse_definition, definition_tags)
        return definitions

    @classmethod
    @safe
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
