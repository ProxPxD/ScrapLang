from argparse import Namespace
from typing import Iterable

from src.constants import Paths, ResourceConstants
from src.context import Context
from src.resouce_managing.short_mem import ShortMemMgr
from src.resouce_managing.valid_data import ValidDataMgr
from src.scrapping import Outcome


class DataGatherer:
    def __init__(self, context: Context):
        self.context: Context = context
        self.valid_args_mgr = ValidDataMgr(Paths.VALID_DATA_FILE, context=self.context)
        self.shor_mem_mgr = ShortMemMgr(Paths.SHORT_MEM_FILE, length=ResourceConstants.SHORT_MEMORY_LENGTH)

    def gather_valid_args(self, scrap_results: Iterable[Outcome]) -> None:
        if self.context.gather_data in ['all', 'ai']:
            self.valid_args_mgr.gather(scrap_results)

    def gather_short_mem(self, parsed: Namespace) -> None:
        if self.context.gather_data in ['all', 'time']:
            self.shor_mem_mgr.add(parsed)