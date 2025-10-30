import shlex
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Callable, Any

import pydash as _
from more_itertools.more import seekable
from requests import Session

from src.cli import CLI
from src.context import Context
from src.logutils import setup_logging
from src.printer import Printer
from src.resouce_managing import ConfMgr
from src.resouce_managing.data_gathering import DataGatherer
from src.scrapping import ScrapMgr
from src.scrapping.core.web_building import get_default_headers


class AppMgr:
    def __init__(self, *, conf_path: Path | str, printer: Callable[[str], Any] = None):
        setup_logging()
        self.conf_mgr = ConfMgr(conf_path)  # TODO: Move paths to context and work from there
        self.context: Context = Context(self.conf_mgr.conf)
        self.data_gatherer = DataGatherer(context=self.context)
        self.cli = CLI(context=self.context, data_gatherer=self.data_gatherer)
        self.scrap_mgr = ScrapMgr()
        self.printer = Printer(context=self.context, printer=printer)

    @contextmanager
    def connect(self) -> Iterator[Session]:
        session = None
        try:
            session = Session()
            session.headers.update(get_default_headers())
            self.scrap_mgr.session = session
            yield session
        finally:
            self.scrap_mgr.session = None
            if session:
                session.close()

    def run(self):
        self.run_single()
        while self.context.loop:
            self.run_single(shlex.split(input()))

    def run_single(self, args: list[str] = None) -> None:
        parsed = self.cli.parse(args)
        if parsed.set or parsed.add or parsed.delete:
            self.context.loop = False
            self.conf_mgr.update_conf(parsed)
            return

        self.context.update(**vars(parsed))
        setup_logging(self.context)
        if self.context.exit and args:
            return
        if self.context.words:
            self.run_scrap()

    def run_scrap(self) -> None:
        # TODO: think when to raise if no word
        with self.connect():
            scrap_results = seekable(self.scrap_mgr.scrap(self.context))
            _.for_each(scrap_results, self.printer.print_result)

        self.conf_mgr.update_lang_order(self.context.all_langs)
        scrap_results.seek(0)
        self.data_gatherer.gather_valid_data(scrap_results)
