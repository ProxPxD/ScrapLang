from dataclasses import dataclass, field, asdict
from functools import cache
from typing import Iterable

from box import Box
from pandas import DataFrame
from requests import Session

from .parsing import ParsedTranslation
from .scrapping import Scrapper
from ..context import Context


# TODO: Everywhere fix hinting

@dataclass(frozen=True)
class BaseDC:

    @classmethod
    @cache
    def all(cls) -> Iterable[str]:
        return asdict(cls()).values()


@dataclass(frozen=True)
class MainScrapKinds(BaseDC):
    INFLECTION: str = 'inflection'
    MAIN_TRANSLATION: str = 'translation'
    INDIRECT_TRANSLATION: str = 'indirect'
    DEFINITION: str = 'definition'


@dataclass(frozen=True)
class HelperScrapKinds(BaseDC):
    SEPERATOR: str = 'seperator'
    NEWLINE: str = '\n'


@dataclass(frozen=True)
class ResultKinds(MainScrapKinds, HelperScrapKinds):
    ...



@dataclass
class ScrapResult:
    kind: str
    args: Box = field(default_factory=Box)  # TODO: think of restricting
    content: DataFrame | Iterable[ParsedTranslation] = None

    def __post_init__(self):
        if self.kind not in ResultKinds().all():
            raise ValueError(f'Result kind is {self.kind}, but expected one of {ResultKinds.all()}')

    def is_fail(self) -> bool:
        return isinstance(self.content, Exception)

    def is_success(self) -> bool:
        return not self.is_fail()


class ScrapMgr:
    def __init__(self, session: Session):
        self.scrapper = Scrapper(session)

    def scrap(self, context: Context) -> Iterable[ScrapResult]:
        for first, last, (from_lang, to_lang, word) in context.grouped_url_triples:
            if is_first_in_group := first and not last:
                group = to_lang if context.groupby == 'lang' else word
                yield ScrapResult(ResultKinds.SEPERATOR, content=group)
            if is_first_to_inflect := context.inflection and first:  # Should take into account grouping method?
                yield self.scrap_inflections(from_lang, word)
            if is_translating := to_lang:
                yield (main := self.scrap_main_translations(from_lang, to_lang, word))
                if context.indirect == 'on' or context.indirect == 'fail' and main.is_fail():
                    yield self.scrap_indirect_translations(from_lang, to_lang, word)
            if context.definition:
                yield self.scrap_definitions(from_lang, word)
            if context.member_sep and context.definition:
                yield ScrapResult(ResultKinds.NEWLINE)

    def scrap_inflections(self, lang: str, word: str) -> ScrapResult:
        return ScrapResult(  # TODO: handle double tables?
            kind=ResultKinds.INFLECTION,
            args=(args := Box(lang=lang, word=word)),
            content=self.scrapper.scrap_inflection(**args)
        )

    def scrap_main_translations(self, from_lang: str, to_lang: str, word: str) -> ScrapResult:
        return ScrapResult(
            kind=ResultKinds.MAIN_TRANSLATION,
            args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word)),
            content=self.scrapper.scrap_main_translations(**args)
        )

    def scrap_indirect_translations(self, from_lang: str, to_lang: str, word: str) -> ScrapResult:
        return ScrapResult(
            kind=ResultKinds.INDIRECT_TRANSLATION,
            args=(args := Box(from_lang=from_lang, to_lang=to_lang, word=word)),
            content=self.scrapper.scrap_indirect_translations(**args)
        )

    def scrap_definitions(self, lang: str, word: str) -> ScrapResult:
        return ScrapResult(
            kind=ResultKinds.DEFINITION,
            args=(args := Box(lang=lang, word=word)),
            content=self.scrapper.scrap_definition(**args)
        )