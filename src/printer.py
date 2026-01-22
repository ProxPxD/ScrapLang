import os
import traceback
from dataclasses import dataclass
from textwrap import indent, wrap
from typing import Any, Callable

import pydash as _
from apscheduler.schedulers.blocking import BlockingScheduler
from pandas import DataFrame
from pydash import chain as c, partial
from tabulate import tabulate
from termcolor import colored

from .context import Context
from .scrapping import Outcome, OutcomeKinds as OK
from .scrapping.wiktio.parsing import Meaning, Pronunciation, WiktioResult

os.environ['TZ'] = 'Europe/Warsaw'

@dataclass(frozen=True)
class Colors:  # TODO: enable conf setting and a flag for no color
    BLUE = (0, 170, 249)
    ORANGE = (247, 126, 0) and 'red'



class Printer:
    def __init__(self, context: Context, printer: Callable[[Any], None] = None, interval: int = 0):
        self.interval = interval
        self.scheduler = BlockingScheduler()
        self._printer = printer or print
        self.context = context

    def print(self, *args, color=None, **kwargs) -> None:
        args = _.map_(args, partial(self.color, color=color))
        self._printer(*args, **kwargs)

    def print_main(self, *args, **kwargs) -> None:
        self.print(*args, **kwargs, color=self.context.color.main)

    def print_secondary(self, *args, **kwargs) -> None:
        self.print(*args, **kwargs, color=self.context.color.pronunciation)

    def print_result(self, outcome: Outcome) -> None:
        match outcome.kind:
            case OK.MAIN_GROUP_SEPERATOR:
                self.print_separator(outcome.results, '━')  # TODO: add as configurable together with numbers
            case OK.SUBGROUP_SEPERATOR:
                self.print_separator(outcome.results, '─')  # TODO: add as configurable together with numbers
            case OK.INFLECTION | OK.GRAMAMR:
                self.print_inflection(outcome)
            case OK.MAIN_TRANSLATION:
                self.print_translation(outcome)
            case OK.INDIRECT_TRANSLATION:
                if outcome.is_success():
                    self.print_translation(outcome)
            case OK.DEFINITION:
                self.print_definitions(outcome)
            case OK.NEWLINE:
                self.print('')
            case OK.WIKTIO:
                self._print_wktio(outcome)
            case _:
                raise ValueError(f'Unknown scrap kind: {outcome.kind}')

    def color(self, to_color, color: str | tuple[int, int, int]) -> str:
        if not color:
            return to_color
        return colored(to_color, tuple(color) if not isinstance(color, str) else color)

    def print_separator(self, group: str, sep: str) -> None:
        bias = len(group)
        colored_group = self.color(group, self.context.color.main)
        self.print(f'{sep*4} {colored_group} {sep*(36-bias)}{sep*4}')

    def print_inflection(self, outcome: Outcome) -> None:
        if outcome.is_fail():
            self.print(outcome.results.args[0])
            return
        match outcome.results:
            case DataFrame() as table: self.print_inflection_table(table)
            case list(): self.print_grammar(outcome)
            case _: raise ValueError(f'Unexpected result type for inflection: {type(outcome.results)}')

    def print_inflection_table(self, table: DataFrame) -> None:
        table_str = tabulate(table, tablefmt='rounded_outline')
        if (olen := len(table_str.split('\n', 1)[0])) > 128:
            table = table.map(lambda x: "\n".join(wrap(x, width=16)))
            table_str = tabulate(table, tablefmt='rounded_grid')
        self.print(table_str)
        if not any((self.context.definition, self.context.inflection)):
            self.print('')

    def print_grammar(self, outcome: Outcome) -> None:
        colored_prefix = self.color(f'{outcome.args.word}: ', self.context.color.main)
        example_string = self.get_grammar_string(outcome.results)
        self.print(f'{colored_prefix}{example_string}')

    def get_grammar_string(self, example_batch: list[list[str]]) -> str:
        match len(example_batch):
            case 1: return ", ".join(example_batch[0])
            case _: return '\n' + '\n'.join(f'  - {", ".join(example_batch)}' for example_batch in example_batch)

    def print_translation(self, outcome: Outcome) -> bool:
        prefix: str = self.get_translation_prefix(outcome)
        colored_prefix = self.color(prefix, self.context.color.main)
        translation_row = self.create_translation_row(outcome)
        self.print(f'{colored_prefix}{translation_row}')
        return outcome.is_success()

    def get_translation_prefix(self, outcome: Outcome) -> str:
        match outcome.kind:
            case OK.MAIN_TRANSLATION: return self.get_member_prefix(outcome)
            case OK.INDIRECT_TRANSLATION: return ' ' * 4 if outcome.is_success() else ''
            case _: raise ValueError(f'Unexpected transltation type: {outcome.kind}')

    def get_member_prefix(self, outcome: Outcome):
        match len(self.context.from_langs):
            case 1: return f'{outcome.args[self.context.member_prefix_arg]}: '
            case _: return f'{outcome.args.word}: '

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
            self.print(outcome.results.args[0])
            return False
        wiktio: WiktioResult = outcome.results
        front = f'{self.color(wiktio.word, self.context.color.main)}: ' if not self.context.to_langs else ''
        wide_front = f'{front}{self._create_wiktio_meaning(wiktio)}'
        if len(wide_front) > 1:  # Not only a newline
            self.print(wide_front)
        more = self._create_wiktio_meanings(wiktio.meanings)
        self.print(more)
        return True

    def _create_wiktio_meaning(self, meaning: Meaning) -> str:
        head, foot = [], []
        p_head, p_foot = self._create_wiktio_pronunciations(meaning.pronunciations)
        head.append(p_head)
        foot.append(p_foot)
        head.append(self._create_wiktio_rel_data(meaning.rel_data))
        foot.append(self._create_wiktio_etymology(meaning.etymology))
        spaced_head = ' '.join(filter(bool, head))
        spaced_foot = '\n'.join(filter(bool, foot))
        return '\n'.join(filter(bool, (spaced_head, spaced_foot)))

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
            self.print(outcome.results.args[0])
            return
        pot_newline = ('', '\n')[bool(self.context.to_langs)]
        ending = f' of "{outcome.args.word}"' if not self.context.to_langs else ''
        self.print(f'{pot_newline}Definitions{ending}:')
        for defi in outcome.results:
            defi_row = f'- {defi.text}{":" if defi.examples else ""}'
            self.print(defi_row)
            for example in defi.examples:
                self.print(f'   - {example}')
