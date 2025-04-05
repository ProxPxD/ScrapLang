from contextlib import contextmanager
from typing import Iterator

from requests import Response, Session
from returns.result import Result

from .context import Context
from .parsing import TranslationParser_, ConjugationParser
from .scrapping import Scrapper, Scrapper_
from .web_pather import GlosbePather, get_default_headers


class AppManager:
    def __init__(self, context: Context = None):
        self.context: Context = context
        self.scrapper = Scrapper()

    @contextmanager
    def connect(self) -> Iterator[Session]:
        session = None
        try:
            session = Session()
            session.headers.update(get_default_headers())
            yield session
        finally:
            session.close()

    def run(self, context: Context = None):
        context = context or self.context
        if not context:
            raise ValueError('No context provided!')
        # TODO: think when to raise if no word

        results = []
        with self.connect() as session:
            scrapper = Scrapper_(session)
            if context.conj:
                conjs = []
                for lang, word in context.source_pairs:
                    conj = scrapper.scrap_conjugation(lang, word)
                    conjs.append(conj)
                for tables in conjs:
                    for table in tables.unwrap():
                        print(table)

            # TODO: Object defining info about what is it?
            for from_lang, to_lang, word in context.url_triples:
                one_page_translation = scrapper.scrap_translation(from_lang, to_lang, word)
                results.append(one_page_translation)
            print('Result:')
            for i, one_page_translation in enumerate(results):
                print(f'Page {1+i}')
                print(type(one_page_translation))
                for record in one_page_translation:
                    print(type(record))
                    print(list(record))

