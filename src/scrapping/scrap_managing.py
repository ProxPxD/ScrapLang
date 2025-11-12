from dataclasses import dataclass, field, asdict
from functools import cache
from typing import Iterable, Optional

from box import Box
from pandas import DataFrame
from requests import Session

from .core.parsing import Result
from .core.scrap_adapting import ScrapAdapter
from .glosbe.scrap_adapting import GlosbeScrapAdapter
from .wiktio.scrap_adapting import WiktioScrapAdapter
from ..context import Context


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
    SEPERATOR: str = 'seperator'
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


class ScrapMgr:
    def __init__(self, session: Session = None):
        self.glosbe_scrapper = GlosbeScrapAdapter()
        self.wiktio_scrapper = WiktioScrapAdapter()
        self.session = session

    @property
    def session(self) -> Session:
        return self._session

    @session.setter
    def session(self, session: Optional[Session]) -> None:
        self._session = session
        for scrapper in self.scrappers:
            scrapper.session = session

    @property
    def scrappers(self) -> Iterable[ScrapAdapter]:
        return self.glosbe_scrapper, self.wiktio_scrapper

    def scrap(self, context: Context) -> Iterable[Outcome]:
        for scrap_it in context.iterate_args():
            from_lang, to_lang, word = scrap_it.args
            if scrap_it.is_first_in_group():
                group = to_lang if context.groupby == 'lang' else word
                yield Outcome(OutcomeKinds.SEPERATOR, results=group)
            if scrap_it.is_at_inflection():
                yield self.scrap_inflections(from_lang, word)
            if scrap_it.is_translating():
                yield (main := self.scrap_main_translations(from_lang, to_lang, word))
                if context.indirect == 'on' or context.indirect == 'fail' and main.is_fail():
                    yield self.scrap_indirect_translations(from_lang, to_lang, word)
            if scrap_it.is_at_wiktio():
                yield self.scrap_wiktio(from_lang, word, context)
            if scrap_it.is_at_definition():
                yield self.scrap_definitions(from_lang, word)
                yield Outcome(OutcomeKinds.NEWLINE)

    def scrap_inflections(self, lang: str, word: str) -> Outcome:
        return Outcome(  # TODO: handle double tables?
            kind=OutcomeKinds.INFLECTION,
            args=(args := Box(lang=lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_inflection(**args)
        )

    def scrap_main_translations(self, from_lang: str, to_lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.MAIN_TRANSLATION,
            args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_main_translations(**args)
        )

    def scrap_indirect_translations(self, from_lang: str, to_lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.INDIRECT_TRANSLATION,
            args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_indirect_translations(**args)
        )

    def scrap_definitions(self, lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.DEFINITION,
            args=(args := Box(lang=lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_definition(**args)
        )

    def scrap_wiktio(self, lang: str, word: str, context: Context = None) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.WIKTIO,
            args=(args := Box(lang=lang, word=word, context=context, frozen_box=True)),
            results=self.wiktio_scrapper.scrap_wiktio_info(**args)
        )