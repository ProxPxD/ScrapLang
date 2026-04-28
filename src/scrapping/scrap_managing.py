from __future__ import annotations

from typing import Iterable, Optional
from typing import TYPE_CHECKING

from box import Box
from requests import Session

from .core.scrap_adapting import ScrapAdapter
from .glosbe.scrap_adapting import GlosbeScrapAdapter
from .outcome import Outcome, OutcomeKinds
from .wiktio.scrap_adapting import WiktioScrapAdapter
from ..context_domain import PrintLevels

if TYPE_CHECKING:
    from src.context import Context


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
        for group in context.iterate_grouped_args():
            from_lang, to_lang, word = group.args
            if group.is_first_in_main_level():
                yield Outcome(OutcomeKinds.get_main_separator(context), results=group.get_header_label(PrintLevels.MAIN))
            if group.is_first_in_mid_level():
                yield Outcome(OutcomeKinds.SUBGROUP_SEPERATOR, results=group.get_header_label(PrintLevels.MID))
            if group.is_first_in_unit_level():
                yield Outcome(OutcomeKinds.SUBGROUP_SEPERATOR, results=group.get_header_label(PrintLevels.UNIT))
            if group.is_first_at_from_inflection():
                yield self.scrap_inflections(from_lang, word)
            if group.is_first_at_from_grammar():
                yield self.scrap_grammar(from_lang, word)
            main = None
            if group.is_translating():
                main = self.scrap_main_translations(from_lang, to_lang, word)
                if group.is_first_at_to_inflection(main):  # TODO: test is_success (ex. lubieć -it instead of lubić)
                    yield self.scrap_inflections(to_lang, main.results[0].word)
                if group.is_first_at_to_grammar(main):
                    yield self.scrap_grammar(to_lang, main.results[0].word)
                yield main
                if context.indirect == 'on' or (context.indirect == 'fail' and main.is_fail()):
                    yield self.scrap_indirect_translations(from_lang, to_lang, word)
            if group.is_first_at_from_overview():
                yield self.scrap_wiktio(from_lang, word)
            if group.is_first_at_from_definition():
                yield self.scrap_definitions(from_lang, word)
                yield Outcome(OutcomeKinds.NEWLINE)
            if group.is_translating():
                if group.is_first_at_to_overview(main):
                    yield self.scrap_wiktio(to_lang, main.results[0].word)
                if group.is_first_at_to_definition(main):
                    yield self.scrap_definitions(to_lang, main.results[0].word)

    def scrap_inflections(self, lang: str, word: str) -> Outcome:
        return Outcome(  # TODO: handle double tables?
            kind=OutcomeKinds.INFLECTION,
            args=(args := Box(lang=lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_inflection(**args),
        )

    def scrap_grammar(self, lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.GRAMAMR,
            args=(args := Box(lang=lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_grammar(**args),
        )

    def scrap_main_translations(self, from_lang: str, to_lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.MAIN_TRANSLATION,
            args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_main_translations(**args),
        )

    def scrap_indirect_translations(self, from_lang: str, to_lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.INDIRECT_TRANSLATION,
            args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_indirect_translations(**args),
        )

    def scrap_definitions(self, lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.DEFINITION,
            args=(args := Box(lang=lang, word=word, frozen_box=True)),
            results=self.glosbe_scrapper.scrap_definition(**args),
        )

    def scrap_wiktio(self, lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.WIKTIO,
            args=(args := Box(lang=lang, word=word, frozen_box=True)),
            results=self.wiktio_scrapper.scrap_wiktio_info(**args),
        )
