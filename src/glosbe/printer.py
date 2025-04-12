import os
from typing import Iterable, Any, Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from pandas import DataFrame
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
        success = True
        for result in scrap_results:
            match result.kind:
                case ScrapKinds.SEPERATOR:
                    self.print_separator(result.content)
                case ScrapKinds.INFLECTION:
                    self.print_inflection(result.content)
                case ScrapKinds.MAIN_TRANSLATION:
                    success = self.print_main_translations(result)
                case ScrapKinds.INDIRECT_TRANSLATION:
                    if self.context.indirect == 'on' or self.context.indirect == 'fail' and not success:
                        self.print_indirect_translations(result)
                case ScrapKinds.DEFINITION:
                    self.print_definitions(result.content)
                case ScrapKinds.NEWLINE:
                    self.printer('')
                case _: raise ValueError(f'Unknown scrap kind: {result.kind}')

    def print_separator(self, group: str) -> None:
        sep = '-'  # TODO: add as configurable together with numbers
        self.printer(f'{sep*8} {group} {sep*25}')

    def print_inflection(self, table: DataFrame) -> None:
        self.printer(tabulate(table, tablefmt='rounded_outline'))
        if not any((self.context.definition, self.context.inflection)):
            self.printer('')

    def print_main_translations(self, result: ScrapResult) -> bool:
        prefix: str = result.args[self.context.prefix_type]
        match result.content:
            case Exception():
                translation_row = result.content.args[0]
                success = False
            case _ if isinstance(result.content, Iterable):
                translation_row = ', '.join(translation.formatted for translation in result.content)
                success = True
            case _: raise ValueError(f'Unexpected main translation content: {result.content}!')
        self.printer(f'{prefix}: {translation_row}')
        return success

    def print_indirect_translations(self, result: ScrapResult) -> None:
        translation_row = ', '.join(translation.formatted for translation in result.content)
        self.printer(f'{" "*4}{translation_row}')

    def print_definitions(self, definitions: Iterable[ParsedDefinition]) -> None:
        self.printer('\nDefinitions:')
        for defi in definitions:
            defi_row = f'- {defi.text}{":" if defi.examples else ""}'
            self.printer(defi_row)
            for example in defi.examples:
                self.printer(f'   - {example}')