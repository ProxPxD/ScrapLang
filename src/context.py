from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import product, repeat
from typing import ClassVar, Iterable, Optional, Any

import pydash as _
from box import Box
from pydash import chain as c

from src.context_domain import ColorSchema, Assume, GroupBy, InferVia, GatherData, Indirect, Mappings, UNSET, \
    Color, color_names
from src.resouce_managing.configuration import Conf


@lambda k: k()
@dataclass(frozen=True)
class Defaults:
    at: str = 'from'
    wiktio: bool = False
    inflection: bool = False
    definition: bool = False
    pronunciation: bool = False

    debug: bool = False
    test: bool = False

    assume: str = 'word'  # TODO: remove
    groupby: str = 'word'
    infervia: str = 'last'
    gather_data: str = 'conf'
    indirect: bool = 'fail'

    color: Color = field(default_factory=lambda: Box(ColorSchema(
        main=(0, 170, 249),
        pronunciation=(247, 126, 0),
    ).model_dump()))

    mappings: Mappings = field(default_factory=dict)
    langs: list = field(default_factory=list)


@dataclass(frozen=False, init=False)
class Context:
    _conf: Conf = None

    words: frozenset[str] = frozenset()
    from_lang: str = None
    to_langs: frozenset[str] = frozenset()

    mapped: tuple[bool] = tuple()

    at: str = UNSET
    wiktio: bool = UNSET
    inflection: bool = UNSET
    definition: bool = UNSET
    pronunciation: bool = UNSET

    debug: bool = UNSET
    test: bool = UNSET

    assume: Assume = UNSET  # TODO: remove
    groupby: GroupBy = UNSET
    infervia: InferVia = UNSET
    gather_data: GatherData = UNSET
    indirect: Indirect = UNSET
    color: Box | Color = UNSET

    loop: bool = False

    mappings: Box | Mappings = UNSET

    _to_filter: ClassVar[set[str]] = {'args', 'reverse', 'add', 'delete', 'set'}

    def __init__(self, conf: Conf):
        self._conf: Conf = conf
        self.update(**conf.model_dump())

    def __getattribute__(self, name: str) -> Any:
        if (val := _.apply_catch(name, super().__getattribute__, [AttributeError], UNSET)) is not UNSET:
            return val
        if (val := getattr(self._conf, name, UNSET)) is not UNSET:
            return val
        if (val := getattr(Defaults, name, UNSET)) is not UNSET:
            return val
        raise AttributeError(f'Attribute "{name}" not found')

    def update(self, **kwargs) -> None:
        kwargs = Box({key: val for key, val in kwargs.items() if key not in self._to_filter})
        if wrong_keys := {key for key in kwargs if not hasattr(self, key)}:
            raise ValueError(f'Context has no such keys: {wrong_keys}')
        for key, val in kwargs.items():
            setattr(self, key, val)

        dict_attrs = _.pick_by(asdict(self), lambda val, key: _.is_dict(val) and not key.startswith('_'))
        for key, val in dict_attrs.items():
            unsets = set(_.get(Defaults, key).keys()) - set(val.keys())
            for subkey in unsets:
                _.set_(self, [key, subkey], _.get(Defaults, [key, subkey]))
            _.set_(self, key, Box(_.get(self, key)))

        self._update_mappings()
        self._update_color()

    def _update_mappings(self) -> None:
        self.mappings = _.map_values(self.mappings, c().apply_if(lambda d: [d], c().is_dict()))

    def _update_color(self) -> None:
        if isinstance(self.color, str):
            self.color = {pos_color: self.color for pos_color in color_names}
        self.color = Box(self.color)

    @property
    def all_langs(self) -> list:
        return [self.from_lang, *self.to_langs]

    @property
    def dest_pairs(self) -> Iterable[tuple[Optional[str], str]]:
        to_langs = self.to_langs or [None]
        match self.groupby:
            case 'lang': return product(to_langs, self.words)
            case 'word': return ((to_lang, word) for word, to_lang in product(self.words, to_langs))
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def source_pairs(self) -> Iterable[tuple[str, str]]:
        return zip(repeat(self.from_lang), self.words)

    @property
    def url_triples(self) -> Iterable[tuple[str, Optional[str], str]]:
        for to_lang, word in self.dest_pairs:
            yield self.from_lang, to_lang, word

    @property
    def n_members(self) -> int:
        match self.groupby:
            case 'lang': return len(self.words)
            case 'word': return len(self.to_langs)
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def n_groups(self) -> int:
        match self.groupby:
            case 'lang': return len(self.to_langs)
            case 'word': return len(self.words)
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def grouped_url_triples(self) -> Iterable:
        for i, (from_lang, to_lang, word) in enumerate(self.url_triples):
            is_first = self._is_first(i)
            is_last = self._is_last(i)
            yield is_first, is_last, (from_lang, to_lang, word)

    def _is_first(self, i) -> bool:
        return self.n_members == 0 or i % self.n_members == 0

    def _is_last(self, i) -> bool:
        return self.n_members == 0 or i % self.n_members == self.n_members - 1

    @property
    def grouparg(self) -> str:
        return f'to_{self.groupby}' if self.groupby == 'lang' else self.groupby

    @property
    def memberarg(self) -> str:
        match self.groupby:
            case 'lang': return 'word'
            case 'word': return 'to_lang'
            case _: raise ValueError(f'Unexpected groupby value: {self.groupby}')

    @property
    def member_prefix_arg(self) -> str:
        match len(getattr(self, f'{self.memberarg}s')):
            case 1: return self.grouparg
            case _: return self.memberarg

    @property
    def exit(self) -> bool:
        return not self.loop

    def is_setting_context(self) -> bool:
        return not self.words and self.loop

    def is_at(self) -> bool:
        return super().__getattribute__('at') is not UNSET

    def is_at_from(self) -> bool:
        return self.at.startswith('f')

    def is_at_to(self) -> bool:
        return self.at.startswith('t')