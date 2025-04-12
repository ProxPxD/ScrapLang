from contextlib import contextmanager
from typing import Iterator

from requests import Session

from .context import Context
from .scrap_managing import ScrapManager
from .scrapping import Scrapper
from .printer import Printer
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
        if not (context := context or self.context):
            raise ValueError('No context provided!')

        with self.connect() as session:
            scrap_results = ScrapManager(session).scrap(context)
            Printer(context).print_all_results(scrap_results)

