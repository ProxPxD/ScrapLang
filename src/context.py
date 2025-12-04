from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import lru_cache
from itertools import product, cycle
from typing import ClassVar, Iterable, Optional, Any, TYPE_CHECKING, Sequence

import pydash as _
from box import Box
from pydash import chain as c

from src.constants import preinitialized
from src.context_domain import ColorSchema, Assume, GroupBy, InferVia, GatherData, Indirect, Mappings, UNSET, \
    Color, color_names, ReanalyzeOn
from src.utils import apply

if TYPE_CHECKING:
    from src.resouce_managing.configuration import Conf

@preinitialized
@dataclass(frozen=True)
class Defaults:
    at: str = 'from'
    wiktio: bool = False
    inflection: bool = False
    definition: bool = False
    pronunciation: bool = False

    debug: bool = False
    test: bool = False

    assume: str = 'lang'  # TODO: remove
    groupby: str = 'word'
    infervia: str = 'last'
    reanalyze_on: ReanalyzeOn = 'gather'
    gather_data: str = 'all'
    indirect: bool = 'fail'

    color: Color = field(default_factory=lambda: Box(ColorSchema(
        main=(0, 170, 249),
        pronunciation=(247, 126, 0),
    ).model_dump()))

    mappings: Mappings = field(default_factory=dict)
    langs: list = field(default_factory=list)



class ScrapIterator:
    def __init__(self, context: Context, i: int, from_lang: str, to_lang: str, word: str, prev: ScrapIterator):
        self._context = context
        self.i: int = i
        self.from_lang: str = from_lang
        self.to_lang: str = to_lang
        self.word: str = word
        self.prev = prev

    def __repr__(self) -> str:
        return f'ScrapIt(i={self.i}, from={self.from_lang}, word={self.word}, to={self.to_lang})'

    @property
    def args(self) -> tuple[str, str, str]:
        return self.from_lang, self.to_lang, self.word

    def _is_first_in_all_main_members(self):
        return self._context.n_all_main_members == 0 or self.i % self._context.n_all_main_members == 0

    @property
    def curr_bundle(self) -> Sequence[str]:
        return self._context.get_from_lang_word_bundle_by_word(self.word)

    @property
    def prev_bundle(self) -> Sequence[str]:
        return self._context.get_from_lang_word_bundle_by_word(self.prev.word)

    def is_in_same_word_bundle_as_prev(self) -> bool:
        return self.prev_bundle != self.curr_bundle

    @apply(on_result=lambda r: print(f'poly-main: {r}'))
    @lru_cache()
    def is_in_poly_main_group(self) -> bool:
        if self._context.n_from_langs == 1:
            return len(self._context.words) > 1 and len(self._context.to_langs) > 1
        match self._context.groupby:
            case 'lang': return len(self._context.to_langs) > 1
            case 'word': return len(self._context.from_lang_word_bundles) > 1
            case _: raise ValueError(f'Unexpected groupby value: {self._context.groupby}')

    def is_first_in_main_group(self) -> bool:
        if self.prev is None:
            return True
        match self._context.groupby:
            case 'lang': return self.prev.to_lang != self.to_lang
            case 'word': return self.is_in_same_word_bundle_as_prev()
            case _: raise ValueError(f'Unexpected groupby value: {self._context.groupby}')

    def is_first_in_poly_main_group(self) -> bool:
        return self.is_in_poly_main_group() and self.is_first_in_main_group()

    def is_in_poly_subgroup(self) -> bool:
        if self._context.n_from_langs == 1:
            return False
        match self._context.groupby:
            case 'lang': return len(self._context.from_lang_word_bundles) > 1
            case 'word': return len(self._context.to_langs) > 1
            case _: raise ValueError(f'Unexpected groupby value: {self._context.groupby}')

    def is_first_in_subgroup(self) -> bool:
        if self.prev is None:
            return True
        match self._context.groupby:
            case 'lang': return self.is_in_same_word_bundle_as_prev()
            case 'word': return self.prev.to_lang != self.to_lang
            case _: raise ValueError(f'Unexpected groupby value: {self._context.groupby}')

    def is_first_in_poly_subgroup(self) -> bool:
        return self.is_in_poly_subgroup() and self.is_first_in_subgroup()

    @property
    def main_group(self) -> str:
        match self._context.groupby:
            case 'lang': return self.to_lang
            case 'word': return self.word_group
            case _: raise ValueError(f'Unexpected groupby value: {self._context.groupby}')

    @property
    def subgroup(self) -> str:
        match self._context.groupby:
            case 'lang': return self.word_group
            case 'word': return self.to_lang
            case _: raise ValueError(f'Unexpected groupby value: {self._context.groupby}')

    @property
    def word_group(self) -> str:
        return '·'.join(self._context.get_from_lang_word_bundle_by_word(self.word))

    @lru_cache()
    def is_last_in_main_group(self) -> bool:
        return self._context.n_all_main_members == 0 or self.i % self._context.n_all_main_members == self._context.n_all_main_members - 1

    @lru_cache()
    def is_at_inflection(self) -> bool:
        return self._context.inflection and self._is_first_in_all_main_members()

    @lru_cache()
    def is_at_translation(self) -> bool:
        return bool(self.to_lang)

    @lru_cache()
    def is_at_wiktio(self) -> bool:
        return self._context.wiktio and self.is_last_in_main_group()

    @lru_cache()
    def is_at_definition(self) -> bool:
        return self._context.definition and self.is_last_in_main_group


@dataclass(frozen=False, init=False)
class Context:
    _conf: Conf = None

    words: tuple[str] = tuple()
    from_langs: tuple[str] = tuple()
    to_langs: tuple[str] = tuple()

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
    indirect: Indirect = UNSET
    color: Box | Color = UNSET
    gather_data: GatherData = UNSET
    infervia: InferVia = UNSET
    reanalyze_on: ReanalyzeOn = UNSET

    loop: bool = False

    mappings: Box | Mappings = UNSET

    _to_filter: ClassVar[set[str]] = {'args', 'reverse', 'add', 'delete', 'set', '_', 'reanalyze'}

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
        return _.interleave(self.from_langs, self.to_langs)

    @property
    def n_from_langs(self) -> int:
        return len(self.from_langs)

    @property
    def from_lang_word_bundles(self) -> Sequence[Sequence[str]]:
        return _.chunk(self.words, self.n_from_langs)

    @property
    def n_sub_members(self) -> int:
        return len(self.from_langs)

    @property
    def n_all_main_members(self) -> int:  # TODO: Theoritically it's a variable value based on the currect bunddle
        match self.groupby:
            case 'lang': return self.n_from_langs * len(list(self.from_lang_word_bundles))
            case 'word': return self.n_from_langs * len(self.to_langs)
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    # @apply(on_result=[lambda r: print(f'n_main_groups: {r}')])
    def n_main_groups(self) -> int:
        match self.groupby:
            case 'lang': return len(self.to_langs)
            case 'word': return len(self.words)
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def dest_pairs(self) -> Iterable[tuple[Optional[str], str]]:
        to_langs = self.to_langs or [None]
        match self.groupby:
            case 'lang': return  ((t, w) for t, ws in product(to_langs, self.from_lang_word_bundles) for w in ws)
            case 'word': return  ((t, w) for ws, t in product(self.from_lang_word_bundles, to_langs) for w in ws)
            case _: raise ValueError(f'Unsupported groupby value: {self.groupby}!')

    @property
    def url_triples(self) -> Iterable[tuple[str, str]]:
        return ((from_lang, *dest) for from_lang, dest in zip(cycle(self.from_langs), self.dest_pairs))

    def iterate_args(self) -> Iterable[ScrapIterator]:
        scrap_it = None
        for i, (from_lang, to_lang, word) in enumerate(self.url_triples):
            scrap_it = ScrapIterator(context=self, i=i, from_lang=from_lang, to_lang=to_lang, word=word, prev=scrap_it)
            yield scrap_it

    def get_from_lang_word_bundle_by_word(self, word: str) -> Sequence[str]:
        n: int = len(self.from_langs)
        i: int = self.words.index(word) // n
        return self.words[i*n:(i+1)*n]

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

    def is_at_from(self) -> bool:
        return self.at.startswith('f') or self.at.startswith('n')

    def is_at_to(self) -> bool:
        return self.at.startswith('t')

