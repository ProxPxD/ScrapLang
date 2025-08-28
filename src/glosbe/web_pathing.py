def get_default_headers():
     return {'User-agent': 'Mozilla/5.0'}
        # {
        #     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        #     'Accept-Encoding': 'gzip, deflate',
        #     'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        #     'Dnt': '1',
        #     'Host': 'httpbin.org',
        #     'Upgrade-Insecure-Requests': '1',
        #     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) '
        #                   'AppleWebKit/537.36 (KHTML, like Gecko) '
        #                   'Chrome/83.0.4103.97 Safari/537.36',
        #     'X-Amzn-Trace-Id': 'Root=1-5ee7bbec-779382315873aa33227a5df6'
        # }


class GlosbePather:
    MAIN_URL: str = 'glosbe.com'

    @classmethod
    def get_word_trans_url(cls, from_lang: str, to_lang: str, word: str) -> str:
        return f'https://{cls.MAIN_URL}/{from_lang}/{to_lang}/{word}'

    @classmethod
    def get_details_url(cls, lang: str, word: str) -> str:
        to_lang = 'en' if lang != 'en' else 'fr'
        return f'{cls.get_word_trans_url(lang, to_lang, word)}/fragment/details'

    @classmethod
    def get_indirect_translations_url(cls, from_lang: str, to_lang: str, word: str) -> str:
        return f'{cls.get_word_trans_url(from_lang, to_lang, word)}/fragment/indirect'