import logging
import shlex
import sys
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator, Optional

from box import Box
from requests import Session

from .cli import CLI
from .configurating import ConfUpdater
from .constants import Paths
from .context import Context
from .printer import Printer
from .scrap_managing import ScrapManager
from .scrapping import Scrapper
from .web_pather import get_default_headers


def setup_logging(context: Context = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    context = Box(asdict(context) if context else {}, default_box=True)
    if context.debug:
        handlers.append(logging.FileHandler(Paths.LOG_DIR))
    logging.basicConfig(
        level=logging.INFO,  # Set minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(levelname)s: %(message)s',
        handlers=handlers,
    )
    warnings.filterwarnings('default' if context.debug else 'ignore')


class AppManager:
    def __init__(self, conf: Box):
        setup_logging()
        self.cli = CLI(conf)
        self.context: Context = Context(conf)
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

    def run(self):
        self.run_single()
        while self.context.loop:
            self.run_single(['t'] + shlex.split(input()))
        ConfUpdater.update_conf(self.context)

    def run_single(self, args: list[str] = None) -> None:
        parsed = self.cli.parse(args)
        # new_context = Context(vars(parsed))
        # if new_context.is_setting_context():
        #     self.context.absorb_context(new_context)
        # else:
        #     new_context.absorb_context(self.context)
        #     parsed = self.cli.process_parsed(parsed)
        #     new_context = Context(vars(parsed))
        self.context = Context(vars(parsed), asdict(self.context))
        setup_logging(self.context)
        if self.context.exit and args:
            return
        if self.context.words:
            self.run_scrap()

    def run_scrap(self) -> None:
        # TODO: think when to raise if no word

        with self.connect() as session:
            scrap_results = ScrapManager(session).scrap(self.context)
            Printer(self.context).print_all_results(scrap_results)
