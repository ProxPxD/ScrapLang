import logging
import shlex
import sys
from argparse import Namespace
from pathlib import Path

import pydash as _
from pydash import chain as c

from src.context import Context
from src.exceptions import InvalidExecution
from src.input_managing.cli import CLI
from src.input_managing.processing import InputProcessor
from src.lang_detecting.preprocessing.data import DataProcessor
from src.resouce_managing.data_gathering import DataGatherer



class InputMgr:
    def __init__(self,
                 context: Context,
                 data_gatherer: DataGatherer = None,
                 data_processor: DataProcessor = None,
                 ):
        self.context = context
        self.cli = CLI(context)
        self.processor = InputProcessor(context, data_processor=data_processor)
        self.data_gatherer = data_gatherer or DataGatherer(context)


    def ingest_input(self, args: list[str] | str = None):
        args = _.apply_if(args, shlex.split, _.is_string) or sys.argv[1:]
        args = _.flat_map(args, c().split('\xa0'))
        parsed = self.cli.parse(args)
        if parsed.reanalyze:  # TODO: test flag with(out) exiting
            logging.debug('Reanalyzing')
            self.processor.data_processor.generate_script_summary()
            if not parsed.words:
                logging.debug('No words for scrapping, exiting after analysis')
        elif not parsed.words and not (parsed.set or parsed.add or parsed.delete):
            raise InvalidExecution('No word specified!')
        parsed = self.processor.process(parsed)
        self.context.update(**vars(parsed))
        return parsed
