import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Any, Sequence, Iterator

import requests
import requests.exceptions as request_exceptions
from requests import Session
from returns.result import safe, Result, Failure

from .parsing import TranslationParser, Record, WrongStatusCodeError, ConjugationParser, AbstractParser, DefinitionParser, WordInfoParser, ParserResult, TranslationParser_, Parser, \
    ParsingException
from .web_pather import get_default_headers, GlosbePather
from requests.exceptions import HTTPError

# TODO: separate
@dataclass(frozen=True)
class TranslationTypes:
    SINGLE = 'Single'
    LANG = 'Lang'
    WORD = 'Word'
    DOUBLE = 'Double'
    CONJ = 'Conjugation'
    DEF = 'Definition'
    WORD_INFO = 'Word Info'


@dataclass(frozen=True)
class WebConstants:
    MAIN_URL = "glosbe.com"


@dataclass
class TransArgs:
    from_lang: str = ''
    to_lang: str = ''
    word: str = ''

    def to_url(self) -> str:
        if not self:
            raise TranslatorArgumentException(self)
        return f'https://{WebConstants.MAIN_URL}/{self.from_lang}/{self.to_lang}/{self.word}'

    def __bool__(self):
        return all(filter(bool, (self.from_lang, self.to_lang, self.word)))


class TranslatorArgumentException(ValueError):
    def __init__(self, trans_args: TransArgs):
        self.trans_args = trans_args


@dataclass
class TranslationResult:
    trans_args: TransArgs = field(default_factory=lambda: TransArgs())
    records: Iterable[Record] = field(default_factory=lambda: [])
    kind: str = TranslationTypes.SINGLE


@dataclass(frozen=True)
class PageCodeMessages:
    PLEASE_REPORT = 'Please, report the case'
    UNHANDLED_PAGE_FULL_MESSAGE = 'Unhandled page code: {}! ' + PLEASE_REPORT
    PAGE_NOT_FOUND_404 = 'Page has not been found (404). Please, check the arguments: {}'  # if the command is correct. Words: {}, glosbe from: , glosbe to:
    PAGE_NOT_FOUND_303 = 'The page has to be redirected (303). ' + PLEASE_REPORT


class ErrorMessages:
    NO_TRANSLATION = 'No translation has been found. Either the arguments were invalid or the requested translation does not exist so far'
    INVALID_ARGUMENT = 'Error! An argument has not been set {}'
    EXCEEDED_NUMBER_OF_RETRIES = 'Error! Exceeded maximum number of retries - check your network connection'
    CONNECTION_ERROR = 'Connection error! Check your network connection'


def get_product(firsts, seconds, by_seconds=False):
    if by_seconds:
        return map(tuple, map(reversed, product(seconds, firsts)))
    return product(firsts, seconds)




class AbstractScrapper:
    def __init__(self, parser: AbstractParser, **kwargs):
        self._parser: AbstractParser = parser
        self.session: requests.sessions.Session | None = None


class TranslatorScrapper(AbstractScrapper):
    def __init__(self, **kwargs):
        super().__init__(parser=TranslationParser(), **kwargs)
        self._word_info_parser = WordInfoParser()
        self.kind: str = ''

    def translate(self, from_lang: str, to_langs: list[str], words: list[str], by_word=False, show_info=True) -> Iterable[TranslationResult]:
        langs_words = get_product(to_langs, words, by_word)

        gathered_info = []
        for to_lang, word in langs_words:
            is_first = show_info and word not in gathered_info
            yield from self.translate_single(from_lang, to_lang, word, is_first=is_first)
            if is_first:
                gathered_info.append(word)

    def translate_single(self, from_lang: str, to_lang: str, word: str, is_first=False) -> TranslationResult:
        trans_args = TransArgs(from_lang, to_lang, word)
        try:
            if is_first:
                word_info_record = self._parse_word_info(trans_args)
                yield TranslationResult(trans_args, word_info_record, kind=TranslationTypes.WORD_INFO)
            records = self._translate_from_url(trans_args)
        except TranslatorArgumentException:
            logging.exception(f'Exception: Invalid argument {str(trans_args)}')
            records = [Record(ErrorMessages.INVALID_ARGUMENT.format(str(trans_args)))]
        yield TranslationResult(trans_args, records, kind=self.kind)

    def request_and_set_page(self, trans_args: TransArgs):
        page: requests.Response = self.session.get(trans_args.to_url(), allow_redirects=True)
        self._parser.set_page(page)
        self._word_info_parser.set_page(page)

    def _parse_word_info(self, trans_args: TransArgs):
        try:
            self.request_and_set_page(trans_args)  # due to laziness needs to be here
            yield from self._word_info_parser.parse()
        except WrongStatusCodeError as err:
            logging.error(f'{err.page.status_code}: {err.page.text}')
            yield Record(self._get_status_code_message(err, trans_args))

    def _translate_from_url(self, trans_args: TransArgs) -> Iterable[Record]:
        try:
            self.request_and_set_page(trans_args)  # due to laziness needs to be here
            yield from self._parser.parse()
        except WrongStatusCodeError as err:
            logging.error(f'{err.page.status_code}: {err.page.text}')
            yield Record(self._get_status_code_message(err, trans_args))
        except request_exceptions.ConnectionError as err:
            logging.exception(traceback.format_exc())
            yield Record(ErrorMessages.CONNECTION_ERROR)
        except TranslatorArgumentException:
            raise TranslatorArgumentException

    def _get_status_code_message(self, err: WrongStatusCodeError, trans_args: TransArgs) -> str:
        match err.page.status_code:
            case 404:
                return PageCodeMessages.PAGE_NOT_FOUND_404.format(str(trans_args))
            case 303:
                return PageCodeMessages.PAGE_NOT_FOUND_303
            case _:
                return PageCodeMessages.UNHANDLED_PAGE_FULL_MESSAGE.format(err.page.status_code)


class ConjugationScrapper(AbstractScrapper):

    def __init__(self, **kwargs):
        super().__init__(parser=ConjugationParser(), **kwargs)

    def get_conjugation(self, lang: str, word: str) -> Iterable:
        trans_args = TransArgs(lang, 'en' if lang != 'en' else 'es', word)
        page: requests.Response = self.session.get(f'{trans_args.to_url()}/fragment/details', allow_redirects=True)
        self._parser.set_page(page)
        yield from self._parser.parse()


class DefinitionScrapper(AbstractScrapper):
    def __init__(self, **kwargs):
        super().__init__(parser=DefinitionParser(), **kwargs)

    def scrap_definitions(self, lang: str, word: str) -> Iterable:
        trans_args = TransArgs(lang, lang, word)
        page: requests.Response = self.session.get(trans_args.to_url(), allow_redirects=True)
        self._parser.set_page(page)
        yield from self._parser.parse()


# TODO: unused
class WordScrapper(AbstractScrapper):
    def __init__(self, **kwargs):
        super().__init__(parser=WordInfoParser(), **kwargs)

    def scrap(self, from_lang, to_lang, word):
        trans_args = TransArgs(from_lang, to_lang, word)
        page: requests.Response
        page = self.session.get(trans_args.to_url(), allow_redirects=True)
        if not page.ok and 'en' not in (from_lang, to_lang):
            trans_args = TransArgs(from_lang, 'en', word)
            page = self.session.get(trans_args.to_url(), allow_redirects=True)
        self._parser.set_page(page)
        yield from self._parser.parse()


####



class TranslationScrapper:
    def __init__(self, session: Session):
        self.session = session

    def scrap(self, from_lang, to_lang, word):
        raise NotImplementedError


    def scrap_(self, from_lang: str, to_langs: list[str], words: list[str], by_word=False, show_info=True) -> Iterable[ParserResult]:
        page: requests.Response = self.session.get(trans_args.to_url(), allow_redirects=True)
        TranslationParser_.parse()



class Scrapper:
    def __init__(self):
        self.args = TransArgs()
        self._translation_scrapper = TranslatorScrapper()
        self._conjugation_scrapper = ConjugationScrapper()
        self._definition_scrapper = DefinitionScrapper()
        self._word_info_scrapper = WordScrapper()

    @property
    def scrappers(self) -> Sequence[AbstractScrapper]:
        return self._translation_scrapper, self._conjugation_scrapper, self._definition_scrapper, self._word_info_scrapper

    @contextmanager
    def connect(self) -> Iterator[Session]:
        session = None
        try:
            session = Session()
            session.headers.update(get_default_headers())
            for scrapper in self.scrappers:
                scrapper.session = session
            yield session
        finally:
            if session:
                session.close()

    def scrap_translation(self, from_lang: str, to_langs: list[str], words: list[str], by_word=False, show_info=True) -> Iterable[TranslationResult]:
        with self.connect():
            yield from self._translation_scrapper.translate(from_lang, to_langs, words, by_word=by_word, show_info=show_info)

    def scrap_conjugation(self, lang: str, word: str) -> Iterable:
        with self.connect():
            yield from self._conjugation_scrapper.get_conjugation(lang, word)

    def scrap_translation_and_conjugation(self, from_lang: str, to_lang: str, word: str, **scrapper_kwargs) -> Iterable[TranslationResult] | Any:
        with self.connect():
            yield self._translation_scrapper.translate(from_lang, to_lang, word, **scrapper_kwargs)
            yield self._conjugation_scrapper.get_conjugation(from_lang, word)

    def scrap_definition(self, lang: str, word: str) -> Iterable:
        with self.connect():
            yield from self._definition_scrapper.scrap_definitions(lang, word)


class Scrapper_:
    def __init__(self, session: Session):
        self.session = session

    @safe
    def scrap(self, url: str, parser: Parser) -> Result[Any, HTTPError | ParsingException]:
        response = self.session.get(url, allow_redirects=True)
        response.raise_for_status()
        return parser.parse(response).unwrap()

    def scrap_translation(self, from_lang: str, to_lang: str, word: str) -> Result[Iterable, HTTPError | ParsingException]:
        url = GlosbePather.get_word_trans_url(from_lang, to_lang, word)
        return self.scrap(url, TranslationParser_)

    def scrap_conjugation(self, lang: str, word: str) -> Result[Any, HTTPError | ParsingException]:
        url = GlosbePather.get_details_url(lang, word)
        return self.scrap(url, ConjugationParser)