import re
from collections import defaultdict
from dataclasses import dataclass, asdict, field, replace
from typing import Iterator, Iterable, Callable, Sequence

from bs4 import PageElement
from bs4.element import Tag
from more_itertools import split_before, last, split_at
from pandas import DataFrame

from ..core.parsing import Result, Parser, with_ensured_tag, ParsingException
from ...constants import supported_languages
import pydash as _
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
    rel_data: dict[str, str]  = field(default_factory=list)
    pronunciations: list[Pronunciation] = None
    etymology: list[str] = None
    inflection: DataFrame = None

@dataclass(frozen=True)
class WiktioResult(Result, Meaning):
    word: str = None
    meanings: list[Meaning] = field(default_factory=list)


class WiktioParser(Parser):
    code_to_wiki: dict = {code: ''.join(last(split_before(descr.split(',')[0], str.isupper, maxsplit=1)))
                          for code, descr in supported_languages.items()}

    @classmethod
    def _split_for_class(cls, tags: Iterable[PageElement], tag_class: str) -> Iterator[list[PageElement]]:
        return split_before(tags, lambda t: isinstance(t, Tag) and tag_class in t.attrs.get('class', []))  # == ['mw-heading', f'mw-heading{n}'])

    @classmethod
    def _filter_for_firsts(cls, batches: Iterable[list[PageElement]], cond: Callable[[str], bool]) -> Iterator[list[PageElement]]:
        return filter(lambda batch: cond(batch[0].text.removesuffix('[edit]')), batches)

    @classmethod
    def _get_target_section_batches(cls, tag: Tag, lang: str) -> dict[str, list[PageElement]]:
        main = tag.select_one('main.mw-body div.mw-body-content div.mw-content-ltr.mw-parser-output')
        # clean_main = cls.filter_to_tags(main.children)
        lang_batches = cls._split_for_class(main, 'mw-heading2')
        target_lang_batch = next(cls._filter_for_firsts(lang_batches, cls.code_to_wiki[lang].__eq__))
        section_batches = cls._split_for_class(target_lang_batch, 'mw-heading')
        section_dict = cls._dictify_section_batches(section_batches)
        return section_dict

    @classmethod
    def _dictify_section_batches(cls, section_batches: Iterable[list[PageElement]]) -> dict[str, list[PageElement]]:
        section_dict, counter = {}, defaultdict(int)
        for batch in section_batches:
            fullname = batch[0].text.removesuffix('[edit]')
            name = re.search(f'(\D+)(\d+)?', fullname).group(1).strip()
            counter[name] += 1
            section_dict[f'{name} {counter[name]}'] = batch
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
    def _filter_section_dict(cls, section_dict: dict[str, list[PageElement]], surf_forms: list[str]) -> dict[str, list[PageElement]]:
        return {key: section for key, section in section_dict.items() if any(key.startswith(form) for form in surf_forms)}

    @classmethod
    def _parse_section(cls, kind: str, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        parse = getattr(cls, f'_parse_{kind}')
        return parse(dc, section)

    @classmethod
    def _parse_pos(cls, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        rel_data_tags = next((tag for tag in section if isinstance(tag, Tag) and tag.name == 'p')).next.children
        outer, brackets = list(split_at(rel_data_tags, lambda t: t.text.strip() == '('))
        outer_tags = filter(lambda t: isinstance(t, Tag) and t.name == 'span', outer)
        outer_feature_dict = {tag.attrs['class'][0]: tag.text for tag in outer_tags}
        feature_bunch = split_at(brackets[:-1], lambda t: t.text.strip() == ',')
        brackets_feature_dict = {name.text: ''.join((e.text for e in val_bunch)).strip() for name, *val_bunch in feature_bunch}
        dc = replace(dc, pos=section[0].text.removesuffix('[edit]'), rel_data={**outer_feature_dict, **brackets_feature_dict})
        return dc

    @classmethod
    def _parse_pronunciation(cls, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        pronunciation_tags = [(tag.select_one('span.ib-content.qualifier-content'), tag.select('span.IPA:not(ul ul span.IPA)'))
                          for tag in cls.filter_to_tags(section[1]) if 'IPA' in tag.text]
        pronunciations = [Pronunciation(name=name_tag.text if name_tag else None, ipas=[ipa_tag.text for ipa_tag in ipa_tags])
                          for name_tag, ipa_tags in pronunciation_tags]
        dc = replace(dc, pronunciations=pronunciations)
        return dc

    @classmethod
    def _parse_etymology(cls, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        content = next((tag for tag in section if tag.name == 'p'))  # Cognate  is next
        etymology_chain = content.text.strip().split(', from')
        fromables = etymology_chain[1:]
        if first_from := etymology_chain[0].startswith('From'):
            fromables.insert(0, etymology_chain[0].removeprefix('From'))
        frommeds = [f'from {fromable}'.removesuffix('.') for fromable in fromables]
        etymology_chain = frommeds if first_from else [etymology_chain[0]] + frommeds
        dc = replace(dc, etymology=etymology_chain)
        return dc

    @classmethod
    def _parse_inflection(cls, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        return dc  # TODO: implemet