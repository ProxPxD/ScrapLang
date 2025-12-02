from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import cache
from typing import Iterable, Optional

from box import Box
from pandas import DataFrame

from .core.parsing import Result


# TODO: Everywhere fix hinting
@dataclass(frozen=True)
class BaseDC:
    @classmethod
    @cache
    def all(cls) -> Iterable[str]:
        return asdict(cls()).values()


@dataclass(frozen=True)
class MainOutcomeKinds(BaseDC):
    MAIN_TRANSLATION: str = 'translation'
    INDIRECT_TRANSLATION: str = 'indirect'
    INFLECTION: str = 'inflection'
    DEFINITION: str = 'definition'
    WIKTIO: str = 'wiktio'


@dataclass(frozen=True)
class HelperOutcomeKinds(BaseDC):
    MAIN_GROUP_SEPERATOR: str = 'main-sep'
    SUBGROUP_SEPERATOR: str = 'sub-sep'
    NEWLINE: str = '\n'


@dataclass(frozen=True)
class OutcomeKinds(MainOutcomeKinds, HelperOutcomeKinds):
    ...


@dataclass
class Outcome:
    kind: str | OutcomeKinds  # Incorect syntax, but there's no right solution
    args: Box = field(default_factory=Box)  # TODO: think of restricting
    results: Optional[DataFrame | Iterable[Result]] = None

    def __post_init__(self):
        if self.kind not in OutcomeKinds().all():
            raise ValueError(f'Outcome kind is {self.kind}, but expected one of {OutcomeKinds.all()}')

    def is_fail(self) -> bool:
        return isinstance(self.results, Exception)

    def is_success(self) -> bool:
        return not self.is_fail()
