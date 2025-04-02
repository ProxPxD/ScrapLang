from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Callable

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString
from bs4.element import Tag


class WrongStatusCodeError(ConnectionError):
    def __init__(self, page: requests.Response, *args):
        super().__init__(*args)
        self.page = page


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


class ConjugationParser(AbstractParser):
    def __init__(self, page: requests.Response = None, **kwargs):
        super().__init__(page, **kwargs)

    def _parse(self):
        try:
            return pd.read_html(self.page.text, keep_default_na=False, header=None)
        except ValueError as e:
            match e.args[0]:
                case "invalid literal for int() with base 10: '100%'":
                    return [pd.DataFrame({'Error': []})]
                case "No tables found":
                    #TODO: implement error printing
                    return [pd.DataFrame({'Error': [e.args[0]]})]
                case _:
                    raise e


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