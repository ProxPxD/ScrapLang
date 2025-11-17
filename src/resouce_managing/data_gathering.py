from argparse import Namespace
from pathlib import Path
from typing import Iterable

from src.constants import Paths, ResourceConstants
from src.context import Context
from src.resouce_managing.short_mem import ShortMemMgr
from src.resouce_managing.valid_data import ValidDataMgr
from src.scrapping import Outcome


class DataGatherer:
    def __init__(self, context: Context, valid_data_file: str | Path = None, short_mem_file: str | Path = None):
        self.context: Context = context
        self.valid_args_mgr = ValidDataMgr(valid_data_file, context=self.context) if valid_data_file else None
        self.shor_mem_mgr = ShortMemMgr(short_mem_file, length=ResourceConstants.SHORT_MEMORY_LENGTH) if short_mem_file else None

    def gather_valid_data(self, scrap_results: Iterable[Outcome]) -> None:
        if self.valid_args_mgr and self.context.gather_data in ['all', 'ai']:
            self.valid_args_mgr.gather(scrap_results)

    def gather_short_mem(self, parsed: Namespace) -> None:
        if self.shor_mem_mgr and self.context.gather_data in ['all', 'time']:
            self.shor_mem_mgr.add(parsed)
