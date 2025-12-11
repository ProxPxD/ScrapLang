from ..core.web_building import UrlBuilder


class WiktioUrlBuilder(UrlBuilder):
    MAIN_URL: str = 'en.wiktionary.org/wiki'
    API_URL: str = 'https://en.wiktionary.org/w/api.php?action=query&titles={word}&prop=revisions&rvprop=content&format=json'

    @classmethod
    def get_wiktio_url(cls, word: str) -> str:
        return f'https://{cls.MAIN_URL}/{word}'

    @classmethod
    def get_wiktio_api_url(cls, word: str) -> str:
        return cls.API_URL.format(word=word)
