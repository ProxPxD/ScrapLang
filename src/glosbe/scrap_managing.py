from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Any

from box import Box
from requests import Session

from .context import Context
from .scrapping import Scrapper


# TODO: Everywhere fix hinting


class ScrapKinds(Enum):
    INFLECTION: str = 'inflection'
    TRANSLATION: str = 'translation'
    DEFINITION: str = 'definition'


@dataclass(frozen=True)
class ScrapResult:
    kind: ScrapKinds
    args: Box  # TODO: think of restricting
    content: Any


class ScrapManager:
    def __init__(self, session: Session):
        self.scrapper = Scrapper(session)

    def scrap(self, context: Context) -> Iterable[ScrapResult]:
        # Todo: should not yield all at once, but order it logically
        if context.inflection:
            yield from self.scrap_inflection(context)
        if context.definition:
            yield from self.scrap_definitions(context)
        if context.to_langs:
            yield from self.scrap_translations(context)

    def scrap_inflection(self, context: Context) -> Iterable[ScrapResult]:
        for lang, word in context.source_pairs:
            yield ScrapResult(  # TODO: handle double tables?
                kind=ScrapKinds.INFLECTION,
                args=(args := Box(lang=lang, word=word)),
                content=self.scrapper.scrap_inflection(**args)
            )

    def scrap_translations(self, context: Context) -> Iterable[ScrapResult]:
        for from_lang, to_lang, word in context.url_triples:
            yield ScrapResult(
                kind=ScrapKinds.TRANSLATION,
                args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word)),
                content=self.scrapper.scrap_translation(**args)
            )

    def scrap_definitions(self, context: Context) -> Iterable[ScrapResult]:
        for lang, word in context.source_pairs:
            yield ScrapResult(
                kind=ScrapKinds.DEFINITION,
                args=(args := Box(lang=lang, word=word)),
                content=self.scrapper.scrap_definition(**args)
            )