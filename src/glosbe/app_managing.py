from contextlib import contextmanager
from typing import Iterator

from requests import Session

from .context import Context
from .parsing import ParsedTranslation
from .scrap_managing import ScrapManager, ScrapKinds, ScrapResult
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
            scrap_results = list(scrap_results)
            i = 0
            for result in scrap_results:
                match result.kind:
                    case ScrapKinds.INFLECTION:
                        for table in result.content:
                            print(table)
                    case ScrapKinds.TRANSLATION:
                        i += 1
                        result: ScrapResult
                        for j, t in enumerate(result.content, 1):
                            t: ParsedTranslation
                            print(
                                f'- {t.word}' +
                                (f' ({t.pos})' if t.pos else '') +
                                (f' [{t.gender}]' if t.gender else '')
                            )
                    case ScrapKinds.DEFINITION:
                        for defi in result.content:
                            print(f'- {defi.text}', end=('', ':\n')[bool(defi.examples)])
                            for example in defi.examples:
                                print(f'   - {example}')
                    case _: raise ValueError(f'Unknown scrap kind: {result.kind}')

