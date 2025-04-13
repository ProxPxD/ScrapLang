from __future__ import annotations

import logging
import shlex
import sys
import warnings
from dataclasses import dataclass, asdict

from box import Box

from glosbe.app_managing import AppManager, Context
from glosbe.cli import CLI
from glosbe.configurating import ConfHandler
from glosbe.constants import Paths


def setup_logging(context: Context = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    context = Box(asdict(context) if context else {}, default_box=True)
    if context.debug:
        handlers.append(logging.FileHandler(Paths.LOG_DIR))
    logging.basicConfig(
        level=logging.INFO,  # Set minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(levelname)s: %(message)s',
        handlers=handlers,
    )
    warnings.filterwarnings('default' if context.debug else 'ignore')


@dataclass(frozen=True)
class ErrorMessages:
    UNKNOWN_EXCEPTION: str = 'Unknown exception occurred!'
    ATTRIBUTE_ERROR: str = 'Error! Please send logs to the creator'


def main():
    setup_logging()
    try:
        # Configurations.init()
        default_conf = ConfHandler.load(Paths.CONF_FILE)
        parsed = CLI(default_conf).parse()
        context = Context(vars(parsed), default_conf)
        setup_logging(context)
        if context.debug:
            logging.basicConfig(level=logging.DEBUG)
        AppManager(context).run()
        #Configurations.change_last_used_languages(*cli.langs)
        #Configurations.save_and_close()
    except:
        raise
    # except AttributeError as err:
    #     logging.error(traceback.format_exc())
    #     TranslationPrinter.out(ErrorMessages.ATTRIBUTE_ERROR)
    # except Exception as ex:
    #     # TODO: in next((flag for flag in self._flags if flag.has_name(name))) of cli parser add an exception to know that the flag has not been added. Similarly in sibling cli_elements
    #     logging.exception(traceback.format_exc())
    #     TranslationPrinter.out(ErrorMessages.UNKNOWN_EXCEPTION, end='\n')


# TODO: test conj
# TODO: test cconj
def get_test_arguments():
    return shlex.split('trans lalka pl en')
    # return shlex.split('trans t dotyczyć pl en')  # exception
    # return shlex.split('trans definicja pl -def')
    # return shlex.split('t dać pl -c')
    # return shlex.split('t sweter pl -c')
    # return shlex.split('t machen de -c')
    # TOOO: add test for "mieć pl de -c"
    # TOOO: add test for "mieć pl -c"
    # TODO: add test for no saved lang exception
    # TODO: add test for mis_tok
    # TODO: add tests for definitions
    # TODO: t anomic en pl


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
