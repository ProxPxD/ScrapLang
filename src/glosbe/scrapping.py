from dataclasses import dataclass
from typing import Iterable, Any, Optional, Callable

from requests import Session, Response
from requests.exceptions import HTTPError
from returns.result import safe, Result

from .parsing import ConjugationParser, TranslationParser_, ParsingException, ParsedTranslation, ParsedDefinition, DefinitionParser_
from .web_pather import GlosbePather


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


# TODO: rename?
class Scrapper:
    def __init__(self, session: Session = None):
        self.session: Optional[Session] = session

    @safe
    def scrap(self, url: str, parse: Callable[[Response], Result]) -> Any | HTTPError | ParsingException:
        response = self.session.get(url, allow_redirects=True)
        response.raise_for_status()
        return parse(response)

    def scrap_translation(self, from_lang: str, to_lang: str, word: str) -> Iterable[ParsedTranslation] | HTTPError | ParsingException:
        url = GlosbePather.get_word_trans_url(from_lang, to_lang, word)
        return self.scrap(url, TranslationParser_.parse)

    def scrap_inflection(self, lang: str, word: str) -> Any | HTTPError | ParsingException:
        url = GlosbePather.get_details_url(lang, word)
        return self.scrap(url, ConjugationParser.parse)

    def scrap_definition(self, lang: str, word: str) -> Iterable[ParsedDefinition] | HTTPError | ParsingException:
        url = GlosbePather.get_word_trans_url(lang, lang, word)
        return self.scrap(url, DefinitionParser_.parse)
