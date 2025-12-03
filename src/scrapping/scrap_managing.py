from __future__ import annotations

from typing import Iterable, Optional
from typing import TYPE_CHECKING

from box import Box
from requests import Session

from .core.scrap_adapting import ScrapAdapter
from .glosbe.scrap_adapting import GlosbeScrapAdapter
from .outcome import Outcome, OutcomeKinds
from .wiktio.scrap_adapting import WiktioScrapAdapter

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
        for scrap_it in context.iterate_args():
            from_lang, to_lang, word = scrap_it.args
            if scrap_it.is_first_in_main_group():
                yield Outcome(OutcomeKinds.get_main_separator(context), results=scrap_it.main_group)
            if scrap_it.is_first_in_subgroup():
                yield Outcome(OutcomeKinds.SUBGROUP_SEPERATOR, results=scrap_it.subgroup)
            if context.is_at_from() and scrap_it.is_at_inflection():
                yield self.scrap_inflections(from_lang, word)
            if scrap_it.is_at_translation():
                main = self.scrap_main_translaztions(from_lang, to_lang, word)
                if context.is_at_to() and scrap_it.is_at_inflection():
                    yield self.scrap_inflections(to_lang, main.results[0].word)
                yield main
                if context.indirect == 'on' or context.indirect == 'fail' and main.is_fail():
                    yield self.scrap_indirect_translations(from_lang, to_lang, word)
                if context.is_at_to() and scrap_it.is_at_wiktio():
                    yield self.scrap_wiktio(to_lang, main.results[0].word)
                if context.is_at_to() and scrap_it.is_at_definition():
                    yield self.scrap_wiktio(to_lang, main.results[0].word)
            if context.is_at_from() and scrap_it.is_at_wiktio():
                yield self.scrap_wiktio(from_lang, word)
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

    def scrap_wiktio(self, lang: str, word: str) -> Outcome:
        return Outcome(
            kind=OutcomeKinds.WIKTIO,
            args=(args := Box(lang=lang, word=word, frozen_box=True)),
            results=self.wiktio_scrapper.scrap_wiktio_info(**args)
        )
