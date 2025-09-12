from ..core.web_building import UrlBuilder


class GlosbeUrlBuilder(UrlBuilder):
    MAIN_URL: str = 'glosbe.com'

    @classmethod
    def get_word_trans_url(cls, from_lang: str, to_lang: str, word: str) -> str:
        return f'https://{cls.MAIN_URL}/{from_lang}/{to_lang}/{word}'

    @classmethod
    def get_details_url(cls, lang: str, word: str) -> str:
        to_lang = ('en', 'fr')[lang == 'en']
        return f'{cls.get_word_trans_url(lang, to_lang, word)}/fragment/details'

    @classmethod
    def get_indirect_translations_url(cls, from_lang: str, to_lang: str, word: str) -> str:
        return f'{cls.get_word_trans_url(from_lang, to_lang, word)}/fragment/indirect'