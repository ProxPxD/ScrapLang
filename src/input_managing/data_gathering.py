import logging
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from src.constants import ResourceConstants
from src.context import Context
from src.input_managing.processing import InputProcessor
from src.lang_detecting.preprocessing.data import DataProcessor
from src.resource_managing.short_mem import ShortMemMgr
from src.resource_managing.valid_data import ValidDataMgr
from src.scrapping import Outcome

if TYPE_CHECKING:
    from src.debug_timer import Timer

class DataGatherer:
    def __init__(self,
            context: Context,
            valid_data_mgr: ValidDataMgr = None,
            short_mem_file: str | Path = None,
            data_processor: DataProcessor = None,
        ):
        self.timer: Timer
        self.context: Context = context
        self.data_processor = data_processor
        self.valid_args_mgr = valid_data_mgr
        self.shor_mem_mgr = ShortMemMgr(short_mem_file, length=ResourceConstants.SHORT_MEMORY_LENGTH) if short_mem_file else None

    def gather_valid_data(self, scrap_results: Iterable[Outcome], processor: InputProcessor) -> None:
        if self.valid_args_mgr and self.context.gather_data in ['all', 'ai']:
            loc = 'GD'
            self.timer.time()
            gathered = self.valid_args_mgr.gather(scrap_results)
            self.timer.time(f'{loc} Gathered')
            if gathered:  # TODO: test
                logging.debug('Retraining after having data gathered')
                self.data_processor.generate_script_summary()
                self.timer.time(f'{loc} Script Generation')
        self.timer.print_all()

    def gather_short_mem(self, parsed: Namespace) -> None:
        if self.shor_mem_mgr and self.context.gather_data in ['all', 'time']:
            self.shor_mem_mgr.add(parsed)
