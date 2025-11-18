import shlex
import sys
from argparse import Namespace
from pathlib import Path

import pydash as _
from pydash import chain as c

from src.context import Context
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
        self.context = context # TODO: convert to using context and data_gatherer instead of conf
        self.cli = CLI(context)
        self.processor = InputProcessor(context, data_processor=data_processor)
        self.data_gatherer = data_gatherer or DataGatherer(context)


    def ingest_input(self, args: list[str] | str = None):
        args = shlex.split(args) if isinstance(args, str) else (args or sys.argv[1:])
        args = _.flat_map(args, c().split('\xa0'))
        parsed = self.cli.parse(args)
        parsed = self.processor.process(parsed)
        self.context.update(**vars(parsed))
        return parsed
