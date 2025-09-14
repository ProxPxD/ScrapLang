from dataclasses import dataclass, asdict, field, replace
from typing import Iterator, Iterable, Callable, Any, Sequence

from bs4.element import Tag
from more_itertools import split_before
from pandas import DataFrame

from ..core.parsing import Result, Parser, with_ensured_tag, ParsingException
from ...constants import supported_languages
import pydash as _
import re
from pydash import chain as c

@dataclass(frozen=True)
class SurfacingEquivalents:
    pos: Sequence[str] = field(default=('Noun', 'Verb', 'Adjective', 'Adverb'))
    pronunciation: Sequence[str] = field(default=('Pronunciation',))
    etymology: Sequence[str] = field(default=('Etymology',))
    inflection: Sequence[str] = field(default=('Declension', 'Conjugation'))

@dataclass(frozen=True)
class Pronunciation:
    ipas: list[str]
    name: str = None

@dataclass(frozen=True)
class Meaning:
    pos: str = None
    relfeats: list[str]  = field(default_factory=list)# Related Features
    pronunciations: list[Pronunciation] = None
    etymology: list[str] = None
    inflection: DataFrame = None

@dataclass(frozen=True)
class WiktioResult(Result, Meaning):
    word: str = None
    meanings: list[Meaning] = field(default_factory=list)


class WiktioParser(Parser):
    code_to_wiki: dict = {code: descr.split(' ')[:2][-1] for code, descr in supported_languages.items()}

    @classmethod
    def _split_for_class(cls, tags: Iterable[Tag], tag_class: str) -> Iterator[list[Tag]]:
        return split_before(tags, lambda t: tag_class in t.attrs.get('class', []))  # == ['mw-heading', f'mw-heading{n}'])

    @classmethod
    def _filter_for_firsts(cls, batches: Iterable[list[Tag]], cond: Callable[[str], bool]) -> Iterator[list[Tag]]:
        return filter(lambda batch: cond(batch[0].text.removesuffix('[edit]')), batches)

    @classmethod
    def _get_target_section_batches(cls, tag: Tag, lang: str) -> dict[str, list[Tag]]:
        main = tag.select_one('main.mw-body div.mw-body-content div.mw-content-ltr.mw-parser-output')
        clean_main = cls.filter_to_tags(main.children)
        lang_batches = cls._split_for_class(clean_main, 'mw-heading2')
        target_lang_batch = next(cls._filter_for_firsts(lang_batches, cls.code_to_wiki[lang].__eq__))
        section_batches = cls._split_for_class(target_lang_batch, 'mw-heading')
        section_dict = {batch[0].text.removesuffix('[edit]'): batch for batch in section_batches}
        return section_dict

    @classmethod
    @with_ensured_tag
    def parse(cls, tag: Tag | str, lang: str) -> WiktioResult | ParsingException:
        section_dict = cls._get_target_section_batches(tag, lang)
        result = WiktioResult()
        for kind, surf_forms in asdict(SurfacingEquivalents()).items():
            surfs = cls._filter_section_dict(section_dict, surf_forms)
            match len(surfs):
                case 0: continue
                case 1:
                    section = list(surfs.values())[0]
                    result = cls._parse_section(kind, result, section)
                case _ as n:
                    while len(result.meanings) < n:
                        result.meanings.append(Meaning())
                    for (i, meaning), section in zip(enumerate(result.meanings), surfs.values()):
                        result.meanings[i] = cls._parse_section(kind, meaning, section)
        return result

    @classmethod
    def _filter_section_dict(cls, section_dict: dict[str, list[Tag]], surf_forms: list[str]):
        return {key: section for key, section in section_dict.items() if any(key.startswith(form) for form in surf_forms)}

    @classmethod
    def _parse_section(cls, kind: str, dc: Meaning | WiktioResult, section: list[Tag]) -> Meaning | WiktioResult:
        parse = getattr(cls, f'_parse_{kind}')
        return parse(dc, section)

    @classmethod
    def _parse_pos(cls, dc: Meaning | WiktioResult, section: list[Tag]) -> Meaning | WiktioResult:
        word_tag, *relfeat_tags =  list(cls.filter_to_tags(next((tag for tag in section if tag.name == 'p')).next.children))
        dc = replace(dc, pos=section[0].text.removesuffix('[edit]'), relfeats=[tag.text for tag in relfeat_tags])
        return dc

    @classmethod
    def _parse_pronunciation(cls, dc: Meaning | WiktioResult, section: list[Tag]) -> Meaning | WiktioResult:
        pronunciation_tags = [(tag.select_one('span.ib-content.qualifier-content'), tag.select('span.IPA:not(ul ul span.IPA)'))
                          for tag in section[1] if 'IPA' in tag.text]
        pronunciations = [Pronunciation(name=name_tag.text, ipas=[ipa_tag.text for ipa_tag in ipa_tags])
                          for name_tag, ipa_tags in pronunciation_tags]
        dc = replace(dc, pronunciations=pronunciations)
        return dc

    @classmethod
    def _parse_etymology(cls, dc: Meaning | WiktioResult, section: list[Tag]) -> Meaning | WiktioResult:
        content = next((tag for tag in section if tag.name == 'p'))  # Cognate  is next
        etymology_chain = content.text.strip().split(', from')
        fromables = etymology_chain[1:]
        if first_from := etymology_chain[0].startswith('From'):
            fromables.insert(0, etymology_chain[0].removeprefix('From'))
        frommeds = [f'from {fromable}'.removesuffix('.') for fromable in fromables]
        etymology_chain = frommeds if first_from else [etymology_chain[0]] + frommeds
        dc = replace(dc, etymology=etymology_chain)
        return dc
