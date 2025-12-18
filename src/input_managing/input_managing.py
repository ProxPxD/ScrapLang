import logging
import shlex
import sys

import pydash as _
from pydash import chain as c

from src.context import Context
from src.context_domain import UNSET
from src.exceptions import InvalidExecution
from src.input_managing.cli import CLI
from src.input_managing.processing import InputProcessor
from src.lang_detecting.preprocessing.data import DataProcessor
from src.input_managing.data_gathering import DataGatherer


class InputMgr:
    def __init__(self,
                 context: Context,
                 data_gatherer: DataGatherer = None,
                 data_processor: DataProcessor = None,
                 ):
        self.context = context
        self.cli = CLI(context)
        self.processor = InputProcessor(context, data_processor=data_processor)


    def ingest_input(self, args: list[str] | str = None):
        args = _.apply_if(args, shlex.split, _.is_string) or sys.argv[1:]
        args = _.flat_map(args, c().split('\xa0'))
        parsed = self.cli.parse(args)
        if parsed.reanalyze:  # TODO: test flag with(out) exiting
            logging.debug('Reanalyzing')
            self.processor.data_processor.generate_script_summary()
            if not parsed.words:
                logging.debug('No words for scrapping, exiting after analysis')
        elif parsed.set or parsed.add or parsed.delete or isinstance(parsed.loop, bool):
            pass
        elif self.context.loop is True and parsed.reverse:
            pass
        elif not parsed.words and self.context.get_only_from_context('loop') is UNSET:  # TODO: test loop not showing message
            raise InvalidExecution('No word specified!')
        parsed = self.processor.process(parsed)

        parsed.from_langs = parsed.from_langs or self.context.from_langs
        parsed.to_langs = parsed.to_langs or self.context.to_langs
        self.context.update(**vars(parsed))
        return parsed
