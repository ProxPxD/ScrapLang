import os
from typing import Iterable, Any, Callable

import pandas
from apscheduler.schedulers.blocking import BlockingScheduler
from tabulate import tabulate

from .context import Context
from .parsing import ParsedDefinition
from .scrap_managing import ScrapResult, ScrapKinds

os.environ['TZ'] = 'Europe/Warsaw'


class Printer:
    def __init__(self, context: Context, printer: Callable[[Any], None] = print, interval: int = 0):
        self.interval = interval
        self.scheduler = BlockingScheduler()
        self.printer = printer
        self.context = context

    def print_all_results(self, scrap_results: Iterable[ScrapResult]) -> None:
        for result in scrap_results:
            match result.kind:
                case ScrapKinds.SEPERATOR:
                    self.print_separator(result.content)
                case ScrapKinds.INFLECTION:
                    self.print_inflection(result.content)
                case ScrapKinds.TRANSLATION:
                    self.print_translations(result)
                case ScrapKinds.DEFINITION:
                    self.print_definitions(result.content)
                case _: raise ValueError(f'Unknown scrap kind: {result.kind}')

    def print_separator(self, group: str) -> None:
        sep = '-'  # TODO: add as configurable together with numbers
        self.printer(f'{sep*8} {group} {sep*25}')

    def print_inflection(self, table: pandas.DataFrame) -> None:
        self.printer(tabulate(table, tablefmt='rounded_outline'))
        if not any((self.context.definition, self.context.inflection)):
            self.printer('')

    def print_translations(self, result: ScrapResult) -> None:
        prefix: str = result.args[self.context.prefix_type]
        translation_row = ', '.join(translation.formatted for translation in result.content)
        self.printer(f'{prefix}: {translation_row}')
        if not self.context.definition:
            self.printer('')

    def print_definitions(self, definitions: Iterable[ParsedDefinition]) -> None:
        self.printer('Definitions:')
        for defi in definitions:
            defi_row = f'- {defi.text}{":" if defi.examples else ""}'
            self.printer(defi_row)
            for example in defi.examples:
                self.printer(f'   - {example}')
        self.printer('')