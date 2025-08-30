import shlex
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator

from box import Box
from requests import Session

from .logutils import setup_logging
from .cli import CLI
from .configurating import ConfUpdater
from .context import Context
from .printer import Printer
from .scrapping import ScrapManager
from .scrapping.web_pathing import get_default_headers


class AppManager:
    def __init__(self, conf: Box):
        setup_logging()
        self.cli = CLI(conf)
        self.context: Context = Context(conf)

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

    def run_single(self, args: list[str] = None) -> None:
        # TODO:
        # To make a proper loop:
        # 0. Context and parsing should be by default uncertain (None)
        # 1. Default infra app context
        # 2. User Conf
        # 3. User Call
        # 4. Loop is a prev context's absorption of the new call based on what's new
        parsed = self.cli.parse(args)
        if parsed.set or parsed.add or parsed.delete:
            self.context.loop = False
            ConfUpdater.update_conf(parsed)
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
        ConfUpdater.update_lang_order(self.context.all_langs)

    def run_scrap(self) -> None:
        # TODO: think when to raise if no word

        with self.connect() as session:
            scrap_results = ScrapManager(session).scrap(self.context)
            Printer(self.context).print_all_results(scrap_results)
