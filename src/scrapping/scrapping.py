from dataclasses import dataclass
from typing import Any, Optional, Callable

from requests import Session, Response
from requests.exceptions import HTTPError

from .parsing import InflectionParser, TranslationParser, ParsingException, ParsedTranslation, ParsedDefinition, \
    DefinitionParser, Parser, CaptchaException
from .web_pathing import GlosbePather


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

    def scrap(self, url: str, parse: Callable[[Response], list[ParsedTranslation | ParsingException] | ParsingException]) -> list[ParsedTranslation] | HTTPError | ParsingException:
        # sleep(6)
        try:
            response = self.session.get(url, allow_redirects=True)
            response.raise_for_status()
        except HTTPError as e:
            return e
        else:
            if Parser.is_captcha(response):
                return CaptchaException('Captcha appeared, robot identified!')
            return parse(response)

    def scrap_main_translations(self, from_lang: str, to_lang: str, word: str) -> list[ParsedTranslation] | HTTPError | ParsingException:
        url = GlosbePather.get_word_trans_url(from_lang, to_lang, word)
        return self.scrap(url, TranslationParser.parse)

    def scrap_indirect_translations(self, from_lang: str, to_lang: str, word: str) -> list[ParsedTranslation] | HTTPError | ParsingException:
        url = GlosbePather.get_indirect_translations_url(from_lang, to_lang, word)
        return self.scrap(url, TranslationParser.parse_indirect_translations)

    def scrap_inflection(self, lang: str, word: str) -> Any | HTTPError | ParsingException:
        url = GlosbePather.get_details_url(lang, word)
        return self.scrap(url, InflectionParser.parse)

    def scrap_definition(self, lang: str, word: str) -> list[ParsedDefinition] | HTTPError | ParsingException:
        url = GlosbePather.get_word_trans_url(lang, lang, word)
        return self.scrap(url, DefinitionParser.parse)
