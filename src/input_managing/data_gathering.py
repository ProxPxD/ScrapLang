import logging
from argparse import Namespace
from pathlib import Path
from typing import Iterable

from src.constants import ResourceConstants
from src.context import Context
from src.input_managing.processing import InputProcessor
from src.lang_detecting.advanced_detecting import AdvancedDetector
from src.lang_detecting.preprocessing.data import DataProcessor
from src.resouce_managing.short_mem import ShortMemMgr
from src.resouce_managing.valid_data import ValidDataMgr
from src.scrapping import Outcome


class DataGatherer:
    def __init__(self,
            context: Context,
            valid_data_mgr: ValidDataMgr = None,
            short_mem_file: str | Path = None,
            data_processor: DataProcessor = None,
        ):
        self.context: Context = context
        self.data_processor = data_processor
        self.valid_args_mgr = valid_data_mgr
        self.shor_mem_mgr = ShortMemMgr(short_mem_file, length=ResourceConstants.SHORT_MEMORY_LENGTH) if short_mem_file else None

    def gather_valid_data(self, scrap_results: Iterable[Outcome], processor: InputProcessor) -> None:
        if self.valid_args_mgr and self.context.gather_data in ['all', 'ai']:
            gathered = self.valid_args_mgr.gather(scrap_results)
            if gathered:  # TODO: test
                logging.debug('Retraining after having data gathered')
                self.data_processor.generate_script_summary()

    def gather_short_mem(self, parsed: Namespace) -> None:
        if self.shor_mem_mgr and self.context.gather_data in ['all', 'time']:
            self.shor_mem_mgr.add(parsed)
