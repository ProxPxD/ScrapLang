import shlex
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock

import pydash as _
from more_itertools.more import seekable
from pydash import chain as c
from requests import Session

from src.conf import ConfFileMgr
from src.context import Context
from src.debug_timer import Timer
from src.exceptions import ScrapLangException
from src.input_managing import InputMgr
from src.input_managing.data_gathering import DataGatherer
from src.lang_detecting.preprocessing.data import DataProcessor
from src.logutils import setup_logging
from src.migration_managing import MigrationManager
from src.printer import Printer
from src.resource_managing.valid_data import ValidDataMgr
from src.scrapping import ScrapMgr
from src.scrapping.core.web_building import get_default_headers


class AppMgr:
    def __init__(self, *,
            conf_path: Path | str,
            valid_data_file: Path | str = None,
            short_mem_file: Path | str = None,
            lang_script_file: Path | str = None,
            printer: Callable[[str], Any] = None,
        ):
        self.timer = Timer(default_new_point=True)
        self.timer.time('App start')
        setup_logging()
        self.timer.time('Log Setup')
        self.conf_mgr = ConfFileMgr(conf_path)  # TODO: Move paths to context and work from there
        self.timer.time('Conf Mgr')
        self.context: Context = Context(self.conf_mgr.conf)
        self.timer.time('context')
        self.valid_data_mgr = ValidDataMgr(valid_data_file, context=self.context) if valid_data_file else None  # TODO: Rework
        self.timer.time('valmgr')
        self.conf_mgr.valid_data_mgr = self.valid_data_mgr
        self.data_processor = DataProcessor(valid_data_mgr=self.valid_data_mgr , lang_script_file=lang_script_file)
        self.timer.time('data processor')
        self.data_gatherer = DataGatherer(context=self.context, valid_data_mgr=self.valid_data_mgr, short_mem_file=short_mem_file, data_processor=self.data_processor)
        self.data_gatherer.timer = self.timer
        self.timer.time('data gatherer')
        # InputMgr time: 0.128
        self.input_mgr = InputMgr(context=self.context, data_processor=self.data_processor)
        self.timer.time('Input Mgr')
        self.scrap_mgr = ScrapMgr()
        self.timer.time('scrap Mgr')
        self.printer = Printer(context=self.context, printer=printer)
        self.timer.time('printer')
        self.migration_mgr = MigrationManager(self.valid_data_mgr)
        self.timer.time('migration mgr')
        self.timer.time('App start')
        self.timer.print_all()
        self.timer.clear()

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

    def run(self) -> None:
        self.timer.time('Pre Run')
        if self.migration_mgr.is_migration_needed():
            self.migration_mgr.migrate()
        self.timer.time('Migration')
        self.run_single()
        while self.context.loop:
            from_langs, to_langs = c().at('from_langs', 'to_langs').map(','.join)(self.context)
            self.printer.print_secondary(f'{from_langs}>{to_langs}❯❯ ', end='')
            self.run_single(shlex.split(input()))

    def run_single(self, args: list[str] = None) -> None:
        try:
            self._raw_run_single(args)
        except ScrapLangException as e:
            msg = e.args[0]
            self.printer.print(msg)

    def _raw_run_single(self, args: list[str] = None) -> None:
        parsed = self.input_mgr.ingest_input(args)
        self.timer.time('Input Ingestion')
        if parsed.set or parsed.add or parsed.delete:
            self.context.loop = False
            self.conf_mgr.update_conf(parsed)
            return

        setup_logging(self.context)
        self.timer.time('Setups and updates')
        if self.context.words:
            self.run_scrap()

    def run_scrap(self) -> None:
        with self.connect():
            scrap_results = seekable(self.scrap_mgr.scrap(self.context))
            _.for_each(scrap_results, self.printer.print_result)
        self.timer.time('Scrapping & Printing')
        self.conf_mgr.update_lang_order(self.context.all_context_langs)
        scrap_results.seek(0)
        self.data_gatherer.valid_args_mgr.timer = self.timer
        self.data_gatherer.gather_valid_data(scrap_results, self.input_mgr.processor)
        self.timer.print_all(clear=True)
