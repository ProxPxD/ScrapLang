from typing import Any

from requests import HTTPError

from .parsing import WiktioParser
from .web_building import WiktioUrlBuilder
from ..core.parsing import ParsingException, Result
from ..core.scrap_adapting import ScrapAdapter
from ...context import Context


class WiktioScrapAdapter(ScrapAdapter):
    def scrap_wiktio_info(self, word: str, lang: str, *, context: Context = None) -> list[Result] | HTTPError | ParsingException:
        url = WiktioUrlBuilder.get_wiktio_url(word)
        results = self.scrap(url, lambda tag: WiktioParser.parse(tag, lang))
        return results
