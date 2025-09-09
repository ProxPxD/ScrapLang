import shlex
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator

from more_itertools.more import seekable, side_effect
from requests import Session
import pydash as _
from pydash import chain as c

from .cli import CLI
from .constants import Paths, ResourceConstants
from .context import Context
from .logutils import setup_logging
from .printer import Printer
from .resouce_managing import ConfMgr
from .resouce_managing.short_mem import ShortMemMgr
from .resouce_managing.valid_data import ValidArgsMgr
from .scrapping import ScrapMgr
from .scrapping.web_pathing import get_default_headers


class AppMgr:
    def __init__(self):
        setup_logging()
        self.conf_mgr = ConfMgr(Paths.CONF_FILE)
        self.cli = CLI(self.conf_mgr.conf, ShortMemMgr(Paths.SHORT_MEM_FILE, length=ResourceConstants.SHORT_MEMORY_LENGTH))
        self.context: Context = Context(self.conf_mgr.conf)
        self.printer = Printer(self.context)
        self.valid_args_mgr = ValidArgsMgr(Paths.VALID_ARGS_FILE, self.context)

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
        with self.connect() as session:
            scrap_results = seekable(ScrapMgr(session).scrap(self.context))
            _.for_each(scrap_results, Printer(self.context).print_result)
        self.conf_mgr.update_lang_order(self.context.all_langs)
        scrap_results.seek(0)
        self.valid_args_mgr.gather(scrap_results)
