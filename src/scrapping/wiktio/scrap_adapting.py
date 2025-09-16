from dataclasses import replace
from doctest import UnexpectedException

from bs4 import Tag
from requests import HTTPError, Response

from .parsing import WiktioParser, WiktioResult
from .web_building import WiktioUrlBuilder
from ..core.parsing import ParsingException, Result
from ..core.scrap_adapting import ScrapAdapter
from ...context import Context


class WiktioScrapAdapter(ScrapAdapter):
    def scrap_wiktio_info(self, word: str, lang: str, *, context: Context = None) -> list[Result] | HTTPError | Exception:
        url = WiktioUrlBuilder.get_wiktio_url(word)
        results = self.scrap(url, self._wrap_parser(word, lang))
        return results

    @classmethod
    def _wrap_parser(cls, word: str, lang: str):
        def parse(tag: Response | Tag) -> WiktioResult | Exception:
            match result := WiktioParser.parse(tag, lang):
                case WiktioResult(): return replace(result, word=word)
                case ParsingException(): return ParsingException(result.args[0] + f' "{word}"')
                case _: return UnexpectedException(f'Unexpected return type "{type(result)}" of "{result}"')
        return parse
