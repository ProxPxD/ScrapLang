from requests import Response
from returns.result import Result

from .context import Context
from .parsing import TranslationParser_
from .scrapping import Scrapper
from .web_pather import GlosbePather


class AppManager:
    def __init__(self, context: Context = None):
        self.context: Context = context
        self.scrapper = Scrapper()

    def run(self, context: Context = None):
        context = context or self.context
        if not context:
            raise ValueError('No context provided!')
        s = Scrapper()
        # TODO: think when to raise if no word

        results = []
        for from_lang, to_lang, word in context.url_triples:
            url = GlosbePather.get_word_trans_url(from_lang, to_lang, word)
            with s.connect() as session:
                # Needa in a layer
                response: Response = session.get(url, allow_redirects=True)
                response.raise_for_status()
                # TODO: Object defining info about what is it?
                one_page_translation = TranslationParser_.parse(response)
                results.append(one_page_translation)
        print('Result:')
        for i, one_page_translation in enumerate(results):
            print(f'Page {1+i}')
            print(type(one_page_translation))
            for record in one_page_translation:
                print(type(record))
                print(list(record))

