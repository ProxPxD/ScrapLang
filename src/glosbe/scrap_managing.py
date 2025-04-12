from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Any

from box import Box
from requests import Session

from .context import Context
from .parsing import ParsedTranslation
from .scrapping import Scrapper


# TODO: Everywhere fix hinting


class ScrapKinds(Enum):  # TODO: think of auto
    INFLECTION: str = 'inflection'
    TRANSLATION: str = 'translation'
    DEFINITION: str = 'definition'
    SEPERATOR: str = 'seperator'


@dataclass(frozen=True)
class ScrapResult:
    kind: ScrapKinds
    args: Box = field(default_factory=Box)  # TODO: think of restricting
    content: Any | ParsedTranslation = None


class ScrapManager:
    def __init__(self, session: Session):
        self.scrapper = Scrapper(session)

    def scrap(self, context: Context) -> Iterable[ScrapResult]:
        for first, last, (from_lang, to_lang, word) in context.grouped_url_triples:
            if first and not last:
                group = to_lang if context.groupby == 'lang' else word
                yield ScrapResult(ScrapKinds.SEPERATOR, content=group)
            if context.inflection and first:
                yield self.scrap_inflections(from_lang, word)
            if to_lang:
                yield self.scrap_translations(from_lang, to_lang, word)
            if context.definition and last:
                yield self.scrap_definitions(from_lang, word)

    def scrap_inflections(self, lang: str, word: str) -> ScrapResult:
        return ScrapResult(  # TODO: handle double tables?
            kind=ScrapKinds.INFLECTION,
            args=(args := Box(lang=lang, word=word)),
            content=self.scrapper.scrap_inflection(**args)
        )

    def scrap_translations(self, from_lang: str, to_lang: str, word: str) -> ScrapResult:
        return ScrapResult(
            kind=ScrapKinds.TRANSLATION,
            args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word)),
            content=self.scrapper.scrap_translation(**args)
        )

    def scrap_definitions(self, lang: str, word: str) -> ScrapResult:
        return ScrapResult(
            kind=ScrapKinds.DEFINITION,
            args=(args := Box(lang=lang, word=word)),
            content=self.scrapper.scrap_definition(**args)
        )