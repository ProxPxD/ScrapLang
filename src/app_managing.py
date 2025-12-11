import shlex
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Callable, Any

import pydash as _
from more_itertools.more import seekable
from requests import Session

from src.context import Context
from src.exceptions import ScrapLangException
from src.input_managing import InputMgr
from src.lang_detecting.preprocessing.data import DataProcessor
from src.logutils import setup_logging
from src.migration_managing import MigrationManager
from src.printer import Printer
from src.resouce_managing import ConfMgr
from src.resouce_managing.data_gathering import DataGatherer
from src.resouce_managing.valid_data import ValidDataMgr
from src.scrapping import ScrapMgr
from src.scrapping.core.web_building import get_default_headers
from pydash import chain as c


class AppMgr:
    def __init__(self, *,
                 conf_path: Path | str,
                 valid_data_file: Path | str = None,
                 short_mem_file: Path | str = None,
                 lang_script_file: Path | str = None,
                 printer: Callable[[str], Any] = None,
                 ):
        setup_logging()
        self.conf_mgr = ConfMgr(conf_path)  # TODO: Move paths to context and work from there
        self.context: Context = Context(self.conf_mgr.conf)
        self.valid_data_mgr = ValidDataMgr(valid_data_file, context=self.context) if valid_data_file else None  # TODO: Rework
        self.migration_mgr = MigrationManager(self.valid_data_mgr)
        self.conf_mgr.valid_data_mgr = self.valid_data_mgr
        self.data_processor = DataProcessor(valid_data_mgr=self.valid_data_mgr , lang_script_file=lang_script_file)
        self.data_gatherer = DataGatherer(context=self.context, valid_data_mgr=self.valid_data_mgr, short_mem_file=short_mem_file, data_processor=self.data_processor)
        self.input_mgr = InputMgr(context=self.context, data_gatherer=self.data_gatherer, data_processor=self.data_processor)
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

    def run(self) -> None:
        if self.migration_mgr.is_migration_needed():
            self.migration_mgr.migrate()
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
        if parsed.set or parsed.add or parsed.delete:
            self.context.loop = False
            self.conf_mgr.update_conf(parsed)
            return

        setup_logging(self.context)
        if self.context.words:
            self.run_scrap()

    def run_scrap(self) -> None:
        with self.connect():
            scrap_results = seekable(self.scrap_mgr.scrap(self.context))
            _.for_each(scrap_results, self.printer.print_result)

        self.conf_mgr.update_lang_order(self.context.all_langs)
        scrap_results.seek(0)
        self.data_gatherer.gather_valid_data(scrap_results)
