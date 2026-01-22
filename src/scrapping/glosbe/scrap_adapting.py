from typing import Any

from requests import HTTPError

from .parsing import TranslationParser, InflectionParser, DefinitionParser
from .web_building import GlosbeUrlBuilder
from ..core.parsing import ParsingException, Result
from ..core.scrap_adapting import ScrapAdapter


class GlosbeScrapAdapter(ScrapAdapter):
    def scrap_main_translations(self, from_lang: str, to_lang: str, word: str) -> list[Result] | HTTPError | ParsingException:
        url = GlosbeUrlBuilder.get_word_trans_url(from_lang, to_lang, word)
        return self.scrap(url, TranslationParser.parse)

    def scrap_indirect_translations(self, from_lang: str, to_lang: str, word: str) -> list[Result] | HTTPError | ParsingException:
        url = GlosbeUrlBuilder.get_indirect_translations_url(from_lang, to_lang, word)
        return self.scrap(url, TranslationParser.parse_indirect_translations)

    def scrap_inflection(self, lang: str, word: str) -> Any | HTTPError | ParsingException:
        url = GlosbeUrlBuilder.get_details_url(lang, word)
        return self.scrap(url, InflectionParser.parse)

    def scrap_grammar(self, lang: str, word: str) -> Any | HTTPError | ParsingException:
        url = GlosbeUrlBuilder.get_details_url(lang, word)
        return self.scrap(url, InflectionParser.parse_grammar)

    def scrap_definition(self, lang: str, word: str) -> list[Result] | HTTPError | ParsingException:
        url = GlosbeUrlBuilder.get_word_trans_url(lang, lang, word)
        return self.scrap(url, DefinitionParser.parse)
