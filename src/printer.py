import os
import random
from textwrap import wrap
from time import sleep
from typing import Iterable, Any, Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from pandas.core.interchange.dataframe_protocol import DataFrame
from tabulate import tabulate
from termcolor import colored

from .context import Context
from .scrapping import ScrapResult, ResultKinds

os.environ['TZ'] = 'Europe/Warsaw'


class Printer:
    def __init__(self, context: Context, printer: Callable[[Any], None] = print, interval: int = 0):
        self.interval = interval
        self.scheduler = BlockingScheduler()
        self.printer = printer
        self.context = context

    def print_result(self, result: ScrapResult) -> None:
        match result.kind:
            case ResultKinds.SEPERATOR:
                self.print_separator(result.content)
            case ResultKinds.INFLECTION:
                self.print_inflection(result)
            case ResultKinds.MAIN_TRANSLATION | ResultKinds.INDIRECT_TRANSLATION:
                self.print_translation(result)
            case ResultKinds.DEFINITION:
                self.print_definitions(result)
            case ResultKinds.NEWLINE:
                self.printer('')
            case _:
                raise ValueError(f'Unknown scrap kind: {result.kind}')

    def print_separator(self, group: str) -> None:
        sep = '-'  # TODO: add as configurable together with numbers
        self.printer(f'{sep*8} {group} {sep*25}')

    def print_inflection(self, result: ScrapResult) -> None:
        if result.is_fail():
            self.printer(result.content.args[0])
            return
        table: DataFrame = result.content
        table_str = tabulate(table, tablefmt='rounded_outline')
        if (olen := len(table_str.split('\n', 1)[0])) > 128:
            table = table.map(lambda x: "\n".join(wrap(x, width=16)))
            table_str = tabulate(table, tablefmt='rounded_grid')
        self.printer(table_str)
        if not any((self.context.definition, self.context.inflection)):
            self.printer('')

    def print_translation(self, result: ScrapResult) -> bool:
        prefix: str = self.get_translation_prefix(result)
        translation_row = self.create_translation_row(result)
        coloured_prefix = colored(prefix, self.context.colour) if self.context.colour != 'no' else prefix
        self.printer(f'{coloured_prefix}{translation_row}')
        return result.is_success()

    def get_translation_prefix(self, result: ScrapResult) -> str:
        match result.kind:
            case ResultKinds.MAIN_TRANSLATION: return f'{result.args[self.context.member_prefix_arg]}: '
            case ResultKinds.INDIRECT_TRANSLATION: return ' '*4 if result.is_success() else ''
            case _: raise ValueError(f'Unexpected transltation type: {result.kind}')

    def create_translation_row(self, result: ScrapResult) -> str:
        match result.is_success():
            case True: return ', '.join(translation.formatted for translation in result.content)
            case False:
                exception = result.content
                if self.context.debug:
                    raise exception
                else:
                    return exception.args[0]
            case _: return ...

    def print_definitions(self, result: ScrapResult) -> None:
        if result.is_fail():
            self.printer(result.content.args[0])
            return
        pot_newline = ('', '\n')[bool(self.context.to_langs)]
        ending = f' of "{result.args.word}"' if not self.context.to_langs else ''
        self.printer(f'{pot_newline}Definitions{ending}:')
        for defi in result.content:
            defi_row = f'- {defi.text}{":" if defi.examples else ""}'
            self.printer(defi_row)
            for example in defi.examples:
                self.printer(f'   - {example}')