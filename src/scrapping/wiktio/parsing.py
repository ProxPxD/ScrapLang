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
    pos: Sequence[str] = field(default=('Pos', 'Noun', 'Verb', 'Adjective', 'Adverb'))
    pronunciation: Sequence[str] = field(default=('Pronunciation',))
    etymology: Sequence[str] = field(default=('Etymology',))
    inflection: Sequence[str] = field(default=('Inflection', 'Declension', 'Conjugation'))

@dataclass(frozen=True)
class Pronunciation:
    ipas: list[str]
    name: str = None

@dataclass(frozen=True)
class Meaning:
    rel_data: dict[str, str]  = field(default_factory=dict)
    pronunciations: list[Pronunciation] = field(default_factory=list)
    etymology: list[str] = field(default_factory=list)
    inflection: DataFrame = None

@dataclass(frozen=True)
class WiktioResult(Result, Meaning):
    word: str = None
    structed_meanings: list[list[Meaning]] = field(default_factory=list)

    @property
    def meanings(self) -> list[Meaning]:
        return _.flatten(self.structed_meanings)

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
        lang_batches = cls._split_for_class(main, 'mw-heading2')
        target_lang_batch = next(cls._filter_for_firsts(lang_batches, cls.code_to_wiki[lang].__eq__))
        section_batches = cls._split_for_class(target_lang_batch, 'mw-heading')
        section_dict = cls._dictify_section_batches(section_batches)
        return section_dict

    @classmethod
    def _dictify_section_batches(cls, section_batches: Iterable[list[PageElement]]) -> dict[str, list[PageElement]]:
        section_dict, counter = {}, defaultdict(int)
        n_sections = 0
        for batch in section_batches:
            fullname = batch[0].text.removesuffix('[edit]')
            name = re.search(f'(\D+)(\d+)?', fullname).group(1).strip()
            name, is_crucial = next(((surf, True) for surf, equivs in asdict(SurfacingEquivalents()).items() if name in equivs), (name, False))
            if not is_crucial:
                continue
            name = name.capitalize()
            if name == 'Etymology':
                n_sections += 1
                counter = defaultdict(int)
            counter[name] += 1
            num = f"{max(n_sections, 1)}.{counter[name]}"
            section_dict[f'{name} {num}'] = batch
        return section_dict

    @classmethod
    @with_ensured_tag
    def parse(cls, tag: Tag | str, lang: str) -> WiktioResult | ParsingException:
        section_dict = cls._get_target_section_batches(tag, lang)
        result = WiktioResult()
        under_surf_mapping = asdict(SurfacingEquivalents())
        for surf, section in section_dict.items():
            under = next((under for under, surfs in under_surf_mapping.items() if any(surf.startswith(s) for s in surfs)))
            digits = surf.rsplit(' ', 1)[-1]
            major, minor = [int(n) for n in re.search(r'(\d+).(\d+)', digits).groups()]
            if len(meanings := result.structed_meanings) < major:
                meanings.append([])
            if len(submeanings := meanings[major-1]) < minor:
                submeanings.append(replace(submeanings[minor-2]) if submeanings else Meaning())
            submeanings[minor-1] = cls._parse_section(under, submeanings[minor-1], section)
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
        rel_data_tags = list(next((tag for tag in section if isinstance(tag, Tag) and tag.name == 'p')).next.children)
        outer, brackets = list(split_at(rel_data_tags, lambda t: t.text.strip() == '(', maxsplit=1))
        outer_tags = filter(lambda t: isinstance(t, Tag) and t.name == 'span', outer)
        outer_feature_dict = {tag.attrs['class'][0]: tag.text for tag in outer_tags}
        feature_bunch = split_at(brackets[:-1], lambda t: t.text.strip() == ',')
        brackets_feature_dict = {name.text: ''.join((e.text for e in val_bunch)).strip() for name, *val_bunch in feature_bunch}
        dc = replace(dc, rel_data={**{'PoS': section[0].text.removesuffix('[edit]')}, **outer_feature_dict, **brackets_feature_dict})
        return dc

    @classmethod
    def _parse_pronunciation(cls, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        pronunciation_tags = [(tag.select_one('span.ib-content.qualifier-content'), tag.select('span.IPA:not(ul ul span.IPA)'))
                          for tag in list(cls.filter_to_tags(section))[1] if 'IPA' in tag.text]
        pronunciations = [Pronunciation(name=name_tag.text if name_tag else None, ipas=[ipa_tag.text for ipa_tag in ipa_tags])
                          for name_tag, ipa_tags in pronunciation_tags]
        dc = replace(dc, pronunciations=pronunciations)
        return dc

    @classmethod
    def _parse_etymology(cls, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        content = next((tag for tag in section if tag.name == 'p' and tag.text))  # Cognate is later
        etymology_chain = _.filter_(content.text.strip().split(', from'))
        if not etymology_chain:
            return dc
        fromables = etymology_chain[1:]
        if first_from := etymology_chain[0].startswith('From'):
            fromables.insert(0, etymology_chain[0].removeprefix('From'))
        frommeds = [f'from {fromable}'.removesuffix('.') for fromable in fromables]
        etymology_chain = frommeds if first_from else [etymology_chain[0]] + frommeds
        dc = replace(dc, etymology=[sent.replace('  ', ' ') for sent in etymology_chain])
        return dc

    @classmethod
    def _parse_inflection(cls, dc: Meaning | WiktioResult, section: list[PageElement]) -> Meaning | WiktioResult:
        return dc  # TODO: implemet