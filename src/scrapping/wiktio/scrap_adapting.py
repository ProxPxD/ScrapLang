from dataclasses import replace

from box import Box
from requests import HTTPError, Response

from .parsing import WiktioParser, WiktioResult
from .web_building import WiktioUrlBuilder
from ..core.parsing import ParsingException, Result
from ..core.scrap_adapting import ScrapAdapter


class WiktioScrapAdapter(ScrapAdapter):
    def scrap_wiktio_info(self, word: str, lang: str) -> list[Result] | HTTPError | Exception:
        params = {
            'action': 'parse',
            'page': word.replace(' ', '_'),
            'format': 'json',
            'prop': 'text',
        }
        url = WiktioUrlBuilder.API_URL
        results = self.scrap(url, self._wrap_parser(word, lang), params=params)
        return results

    def _wrap_parser(self, word: str, lang: str):
        def parse(response: Response) -> WiktioResult | Exception:
            page = Box(response.json(), default_box=True)
            if page.error:
                return ParsingException(page.error.info + f': "{response.url}"')
            match result := WiktioParser.parse(page.parse.text['*'], lang, self):
                case WiktioResult(): return replace(result, word=word)
                case ParsingException(): return ParsingException(result.args[0] + f' "{word}"')
        return parse
