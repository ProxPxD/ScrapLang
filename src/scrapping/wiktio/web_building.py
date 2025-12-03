from ..core.web_building import UrlBuilder


class WiktioUrlBuilder(UrlBuilder):
    MAIN_URL: str = 'en.wiktionary.org/wiki'

    @classmethod
    def get_wiktio_url(cls, word: str) -> str:
        return f'https://{cls.MAIN_URL}/{word}'