from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum
from io import StringIO
from typing import Iterable, Callable

import pandas as pd
import pydash as _
import requests
from bs4 import BeautifulSoup, NavigableString, ResultSet
from bs4.element import Tag
from decorator import decorator
from pydash import chain as c
from requests import Response
from returns.maybe import Maybe
from returns.result import safe, Result, Success, Failure

from .context import Context

is_response = _.is_instance_of_cmp(Response)
is_context = _.is_instance_of_cmp(Context)


class WrongStatusCodeError(ConnectionError):
    def __init__(self, page: requests.Response, *args):
        super().__init__(*args)
        self.page = page


class ParsingException(ValueError):
    pass


@dataclass
class Record:
    translation: str = ''
    part_of_speech: str = ''
    gender: str = ''

    def __bool__(self):
        return any(filter(bool, (self.translation, self.part_of_speech, self.gender)))


@dataclass
class Definition:
    definition: str = ''
    example: str = ''


@dataclass(frozen=True)
class ScrappingKind(StrEnum):
    TRANSLATION: str = 'translation'


@dataclass
class ParserResult:
    kind: ScrappingKind
    context: Context

    translation: str = ''
    part_of_speech: str = ''
    gender: str = ''

    definition: str = ''
    example: str = ''

    exception: Exception = None

    def __bool__(self):
        return c(asdict(self).values()).reject(is_context).some()

    @property
    def is_success(self) -> bool:
        return self.exception is None


class AbstractParser(ABC):
    def __init__(self, page: requests.Response = None, **kwargs):
        super().__init__(**kwargs)
        self.page = page

    def set_page(self, page: requests.Response):
        self.page = page

    def parse(self):
        if not self.page.ok:
            raise WrongStatusCodeError(self.page)
        yield from self._parse()

    @abstractmethod
    def _parse(self):
        raise NotImplemented


class FeatureParser(AbstractParser, ABC):
    def __init__(self, page: requests.Response = None, **kwargs):
        super().__init__(page, **kwargs)

    def _get_create_featured_record_from_tag(self, get_main: Callable[[Tag], str], get_spans: Callable[[Tag], list]):
        return lambda tag: self._create_featured_record_with_spans(get_main(tag), get_spans(tag))

    def _create_featured_record_with_spans(self, main: str, spans: list) -> Record:
        features = self._get_features(spans)
        return Record(main, *features)

    def _get_features(self, spans):
        part_of_speech = self._get_part_of_speech(spans)
        gender = self._get_gender(spans)
        return part_of_speech, gender

    def _get_part_of_speech(self, spans: list[Tag, ...]) -> str:
        return self._get_ith(spans, 0)

    def _get_gender(self, spans: list[Tag, ...]) -> str:
        return self._get_ith(spans, 1)

    def _get_ith(self, items: list[Tag, ...], i: int):
        return items[i].text if len(items) > i else ''


class WordInfoParser(FeatureParser):
    def __init__(self, page: requests.Response = None, **kwargs):
        super().__init__(page, **kwargs)

    def _parse(self) -> Iterable[Record]:  # TODO: add test for it
        soup = BeautifulSoup(self.page.text, features="html.parser")
        word_info_tag = soup.find('div', {'class': 'text-xl text-gray-900 px-1 pb-1'})
        # actual_trans = filter(lambda trans_elem: trans_elem.select_one('h3'), trans_elems)
        get_featured_record = self._get_create_featured_record_from_tag(self._get_word, self._get_spans)
        record = get_featured_record(word_info_tag)
        yield record

    def _get_word(self, tag: Tag) -> str:
        word = tag.select_one('span', {'class': 'font-medium break-words'}).text
        return ''

    def _get_spans(self, tag: Tag) -> list:
        main_span = tag.find('span', {'class': 'text-xxs text-gray-500 inline-block'})
        return main_span.find_all('span')


class TranslationParser(FeatureParser):
    def __init__(self, page: requests.Response = None, **kwargs):
        super().__init__(page, **kwargs)

    def _parse(self) -> Iterable[Record]:
        soup = BeautifulSoup(self.page.text, features="html.parser")
        trans_elems = soup.find_all('div', {'class': 'inline leading-10'})
        actual_trans = filter(lambda trans_elem: trans_elem.select_one('h3'), trans_elems)
        get_featured_record = self._get_create_featured_record_from_tag(self._get_translation, self._get_spans)
        records = map(get_featured_record, actual_trans)
        return records

    def _get_translation(self, translation_tag: Tag) -> str:
        return translation_tag.select_one('h3').text.replace('\n', '')

    def _get_spans(self, translation_tag: Tag) -> list[Tag, ...]:
        main_span = translation_tag.select_one('span', {'class': 'text-xxs text-gray-500'})
        return main_span.find_all('span')



class DefinitionParser(AbstractParser):

    def __init__(self, page: requests.Response = None, **kwargs):
        super().__init__(page, **kwargs)

    def _parse(self):
        soup = BeautifulSoup(self.page.text, features="html.parser")
        definitions_nodes = soup.find_all('li', {'class': 'pb-2'})
        definitions = map(self._parse_definition, definitions_nodes)
        return definitions

    def _parse_definition(self, definition_tag: Tag) -> Definition:
        definition_text = self._parse_definition_text(definition_tag)
        example = self._parse_example(definition_tag)
        return Definition(definition_text, example)

    def _parse_definition_text(self, definition_tag: Tag) -> str:
        core_content = ''.join((content for content in definition_tag.contents if isinstance(content, (NavigableString, str))))
        return core_content\
            .removeprefix('\n')\
            .removesuffix('\n')\
            .replace('\n\n', ' ')\
            .replace('\n', ' ')\
            .removeprefix(' ')\
            .removeprefix(' ')

    def _parse_example(self, definition_tag: Tag) -> str:
        example_tag = definition_tag.select_one('div', {'class': 'border-l-2 pl-2 border-gray-200 text-gray-600 '})
        example = example_tag.text.replace('\n', '') if example_tag else ''
        if any((to_skip == example for to_skip in ('adjective', 'verb', 'noun'))):
            return ''
        return example


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


class TranslationParser_(Parser):
    @classmethod
    def _parse(cls, tag: Tag) -> Iterable[Result[tuple(str, Maybe[str], Maybe[str]), ParsingException]]:
        if not (trans_divs := tag.find_all('div', {'class': 'inline leading-10'})):
            raise ParsingException('No translation div!')
        for trans_div in map(Success, trans_divs):
            word = trans_div.bind(cls._get_translated_word)
            spans: Result = trans_div.bind(cls._get_spans)
            # Assume for now
            pos = spans.map(c().get('0.text'))     # None is not Failure
            gender = spans.map(c().get('1.text'))  # None is not Failure
            yield Result.do((w, Maybe.from_optional(g), Maybe.from_optional(p)) for w in word for g in gender for p in pos)

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
        for table in tag.select('div #grammar_0_0 table'):
            yield pd.read_html(StringIO(str(table)), keep_default_na=False, header=None)
