import os
import traceback
from dataclasses import dataclass
from textwrap import wrap, indent
from typing import Any, Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from pandas.core.interchange.dataframe_protocol import DataFrame
from pydash import chain as c
from tabulate import tabulate
from termcolor import colored

from .context import Context
from .scrapping import Outcome, OutcomeKinds
from .scrapping.wiktio.parsing import WiktioResult, Meaning, Pronunciation

os.environ['TZ'] = 'Europe/Warsaw'

@dataclass(frozen=True)
class Colors:  # TODO: enable conf setting and a flag for no color
    BLUE = (0, 170, 249)
    ORANGE = (247, 126, 0) and 'red'



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

    def color(self, to_color, color: str | tuple[int, int, int]) -> str:
        return colored(to_color, color)

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
        colored_prefix = self.color(prefix, self.context.color.main)
        self.printer(f'{colored_prefix}{translation_row}')
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
        if outcome.is_fail():
            self.printer(outcome.results.args[0])
            return False
        wiktio: WiktioResult = outcome.results
        front = f'{self.color(wiktio.word, self.context.color.main)}: ' if not self.context.to_langs else ''
        wiktio_str = f'{front}{self._create_wiktio_meaning(wiktio)}{self._create_wiktio_meanings(wiktio.meanings)}'
        self.printer(wiktio_str)
        return True

    def _create_wiktio_meaning(self, meaning: Meaning) -> str:
        head, foot = [], []
        p_head, p_foot = self._create_wiktio_pronunciations(meaning.pronunciations)
        head.append(p_head)
        foot.append(p_foot)
        head.append(self._create_wiktio_rel_data(meaning.rel_data))
        foot.append(self._create_wiktio_etymology(meaning.etymology))
        return " ".join(filter(bool, head)) + '\n' + '\n'.join(filter(bool, foot))

    def _create_wiktio_pronunciations(self, pronunciations: list[Pronunciation]) -> tuple[str, str]:
        names = c(pronunciations).map(c().get('name')).filter().value()
        match bool(names):
            case True: return '', 'pronunciation:\n' + '\n'.join(f'  - {p.name}: {", ".join(self.color(ipa, self.context.color.pronunciation) for ipa in p.ipas)}' for p in pronunciations)
            case False: return c(pronunciations).map(c().get('ipas')).flatten().join(', ').value(), ''
            case _: raise Exception('Impossible')

    def _create_wiktio_rel_data(self, rel_data: dict[str, str]) -> str:
        if rel_data:
            return '[' + ', '.join(f'{key}: {val}' if val else key for key, val in rel_data.items()) + ']'
        return ''

    def _create_wiktio_etymology(self, etymology: list[str]) -> str:
        if etymology:
            return indent('etymology:\n' + '\n'.join(f'  - {etymology}' for etymology in etymology), ' '*2)
        return ''

    def _create_wiktio_meanings(self, meanings: list[Meaning]) -> str:
        if meanings:
            return 'meanings:\n' + '\n'.join(indent(f'• {self._create_wiktio_meaning(meaning)}', ' '*2) for meaning in meanings)
        return ''

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