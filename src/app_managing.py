import shlex
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator

import pydash as _
from more_itertools.more import seekable
from requests import Session

from src.cli import CLI
from src.constants import Paths
from src.context import Context
from src.logutils import setup_logging
from src.printer import Printer
from src.resouce_managing import ConfMgr
from src.resouce_managing.data_gathering import DataGatherer
from src.scrapping import ScrapMgr
from src.scrapping.core.web_building import get_default_headers


class AppMgr:
    def __init__(self):
        setup_logging()
        self.conf_mgr = ConfMgr(Paths.CONF_FILE)  # TODO: Move paths to context and work from there
        self.context: Context = Context(self.conf_mgr.conf)
        self.data_gatherer = DataGatherer(context=self.context)
        self.cli = CLI(self.conf_mgr.conf, context=self.context, data_gatherer=self.data_gatherer)
        self.scrap_mgr = ScrapMgr()
        self.printer = Printer(context=self.context)

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
            self.conf_mgr.update_conf(parsed)
        # new_context = Context(vars(parsed))
        # if new_context.is_setting_context():
        #     self.context.absorb_context(new_context)
        # else:
        #     new_context.absorb_context(self.context)
        #     parsed = self.cli.process_parsed(parsed)
        #     new_context = Context(vars(parsed))
        self.context = Context(vars(parsed), asdict(self.context))  # TODO: update context instead of reassigning
        setup_logging(self.context)
        if self.context.exit and args:
            return
        if self.context.words:
            self.run_scrap()
        self.conf_mgr.update_lang_order(self.context.all_langs)

    def run_scrap(self) -> None:
        # TODO: think when to raise if no word
        with self.connect():
            scrap_results = seekable(self.scrap_mgr.scrap(self.context))
            _.for_each(scrap_results, Printer(self.context).print_result)

        self.conf_mgr.update_lang_order(self.context.all_langs)
        scrap_results.seek(0)
        self.data_gatherer.gather_valid_data(scrap_results)
