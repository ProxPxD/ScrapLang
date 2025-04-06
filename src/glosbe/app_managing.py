from contextlib import contextmanager
from typing import Iterator

from requests import Session

from .context import Context
from .scrap_managing import ScrapManager, ScrapKinds
from .scrapping import Scrapper
from .web_pather import get_default_headers


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

        with self.connect() as session:
            scrap_results = ScrapManager(session).scrap(context)
            i = 0
            for result in scrap_results:
                match result.kind:
                    case ScrapKinds.INFLECTION:
                        for table_batch in result.content:
                            for table in table_batch:
                                print(table)
                    case ScrapKinds.TRANSLATION:
                        i += 1
                        for j, record in enumerate(result.content, 1):
                            print(list(record))
                    case ScrapKinds.DEFINITION:
                        for defi in result.content:
                            print(list(defi))
                    case _: raise ValueError(f'Unknown scrap kind: {result.kind}')

