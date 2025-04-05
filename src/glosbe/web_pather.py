from dataclasses import dataclass


class GlosbePather:
    MAIN_URL = "glosbe.com"

    @classmethod
    def get_word_trans_url(cls, from_lang: str, to_lang: str, word: str) -> str:
        return f'https://{cls.MAIN_URL}/{from_lang}/{to_lang}/{word}'