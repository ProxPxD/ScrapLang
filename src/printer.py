import os
import traceback
from dataclasses import asdict
from textwrap import wrap
from typing import Any, Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from pandas.core.interchange.dataframe_protocol import DataFrame
from tabulate import tabulate
from termcolor import colored

from .context import Context
from .scrapping import Outcome, OutcomeKinds
from .scrapping.wiktio.parsing import WiktioResult, Meaning

os.environ['TZ'] = 'Europe/Warsaw'


class Printer:
    def __init__(self, context: Context, printer: Callable[[Any], None] = print, interval: int = 0):
        self.interval = interval
        self.scheduler = BlockingScheduler()
        self.printer = printer
        self.context = context

    def print_result(self, outcome: Outcome) -> None:
        match outcome.kind:
            case OutcomeKinds.SEPERATOR:
                self.print_separator(outcome.results)
            case OutcomeKinds.INFLECTION:
                self.print_inflection(outcome)
            case OutcomeKinds.MAIN_TRANSLATION:
                self.print_translation(outcome)
            case OutcomeKinds.INDIRECT_TRANSLATION:
                if outcome.is_success():
                    self.print_translation(outcome)
            case OutcomeKinds.DEFINITION:
                self.print_definitions(outcome)
            case OutcomeKinds.NEWLINE:
                self.printer('')
            case OutcomeKinds.WIKTIO:  # TODO: WIKTIO, redo
                self._print_wktio(outcome)
            case _:
                raise ValueError(f'Unknown scrap kind: {outcome.kind}')

    def print_separator(self, group: str) -> None:
        sep = '-'  # TODO: add as configurable together with numbers
        self.printer(f'{sep*8} {group} {sep*25}')

    def print_inflection(self, outcome: Outcome) -> None:
        if outcome.is_fail():
            self.printer(outcome.results.args[0])
            return
        table: DataFrame = outcome.results
        table_str = tabulate(table, tablefmt='rounded_outline')
        if (olen := len(table_str.split('\n', 1)[0])) > 128:
            table = table.map(lambda x: "\n".join(wrap(x, width=16)))
            table_str = tabulate(table, tablefmt='rounded_grid')
        self.printer(table_str)
        if not any((self.context.definition, self.context.inflection)):
            self.printer('')

    def print_translation(self, outcome: Outcome) -> bool:
        prefix: str = self.get_translation_prefix(outcome)
        translation_row = self.create_translation_row(outcome)
        coloured_prefix = colored(prefix, self.context.colour) if self.context.colour != 'no' else prefix
        self.printer(f'{coloured_prefix}{translation_row}')
        return outcome.is_success()

    def get_translation_prefix(self, outcome: Outcome) -> str:
        match outcome.kind:
            case OutcomeKinds.MAIN_TRANSLATION: return f'{outcome.args[self.context.member_prefix_arg]}: '
            case OutcomeKinds.INDIRECT_TRANSLATION: return ' ' * 4 if outcome.is_success() else ''
            case _: raise ValueError(f'Unexpected transltation type: {outcome.kind}')

    def create_translation_row(self, outcome: Outcome) -> str:
        match outcome.is_success():
            case True: return ', '.join(trans_result.formatted for trans_result in outcome.results)
            case False: return self._get_stacktrace_or_exception(outcome)
            case _: raise ValueError('Unexpected branching')

    def _get_stacktrace_or_exception(self, outcome: Outcome) -> str:
            exception = outcome.results
            if self.context.debug:
                return traceback.format_exc()
            else:
                return exception.args[0]

    def _print_wktio(self, outcome: Outcome) -> bool:
        wiktio: WiktioResult = outcome.results
        for kind, value in asdict(wiktio).items():
            match value:
                case list():
                    self.printer(f'{kind}: ')
                    for elem in value:
                        self.printer(f'  - {elem}')
                case dict() | Meaning():
                    self.printer(f'{kind}: ')
                    for key, val in value.items():
                        self.printer(f'  - {key}: {val}')
                case _:
                    self.printer(f'{kind}: {value}')
        ...

    # def _print_wiktio_pos(self, ):

    def print_definitions(self, outcome: Outcome) -> None:
        if outcome.is_fail():
            self.printer(outcome.results.args[0])
            return
        pot_newline = ('', '\n')[bool(self.context.to_langs)]
        ending = f' of "{outcome.args.word}"' if not self.context.to_langs else ''
        self.printer(f'{pot_newline}Definitions{ending}:')
        for defi in outcome.results:
            defi_row = f'- {defi.text}{":" if defi.examples else ""}'
            self.printer(defi_row)
            for example in defi.examples:
                self.printer(f'   - {example}')