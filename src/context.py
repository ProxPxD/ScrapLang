from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cache, cached_property
from itertools import product
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Sequence

import pydash as _
from box import Box
from pydash import chain as c

from src.constants import preinitialized
from src.context_domain import (
    UNSET,
    ArgKind,
    Assume,
    Color,
    ColorSchema,
    GatherData,
    GroupBy,
    Indirect,
    InferVia,
    Mappings,
    PrintLevels,
    RetrainOn,
    SpecialEnum,
    color_names,
)

if TYPE_CHECKING:
    from src.conf import Conf
    from src.scrapping import Outcome

@preinitialized
@dataclass(frozen=True)
class Defaults:
    at: str = 'from'
    wiktio: bool = False
    inflection: bool = False
    grammar: bool = False
    definition: bool = False
    pronunciation: bool = False

    debug: bool = False
    test: bool = False

    assume: str = 'lang'
    groupby: GroupBy = GroupBy.WORD
    grouping_3: tuple[ArgKind] = (ArgKind.TO_LANGS, ArgKind.WORDS, ArgKind.FROM_LANGS)
    grouping_2: tuple[ArgKind] = (ArgKind.FROM_LANGS, ArgKind.WORDS, ArgKind.TO_LANGS)
    run_grouping: tuple[ArgKind] = None
    infervia: str = 'last'
    retrain_on: RetrainOn = 'gather'
    retrain: bool = False
    gather_data: str = 'all'
    indirect: bool = 'fail'

    color: Color = field(default_factory=lambda: Box(ColorSchema(
        main=(0, 170, 249),
        pronunciation=(247, 126, 0),
    ).model_dump()))

    mappings: Mappings = field(default_factory=dict)
    langs: list = field(default_factory=list)

    loop: bool = False

class ArgGroup:
    app_order: ClassVar[list[ArgKind]] = [ArgKind.FROM_LANGS, ArgKind.TO_LANGS, ArgKind.WORDS]

    def __init__(self, group: Sequence[str], grouping: list[ArgKind], context: Context):
        self.group = group
        self.grouping = grouping
        self.context = context

    def __repr__(self) -> str:
        args = self.args
        return f'ArgGroup({args=})'

    @cached_property
    def _group_perm(self) -> list[int]:  # TODO: think of moving to context as it's not group specific
        return [self.grouping.index(arg) for arg in self.app_order]

    @property
    def args(self) -> list[str]:
        return [self.group[i] for i in self._group_perm]

    def get_kind(self, level: PrintLevels | int) -> ArgKind:
        i = level if isinstance(level, int) else level.i
        return self.grouping[i]

    def get_arg(self, kind: ArgKind | PrintLevels) -> str:
        kind = self.get_kind(kind) if isinstance(kind, PrintLevels) else kind
        i = self.grouping.index(kind)
        return self.group[i]

    @property
    def main_kind(self) -> ArgKind:
        return self.get_kind(PrintLevels.MAIN)

    @property
    def main_arg(self) -> str:
        return self.get_arg(self.main_kind)

    @property
    def sub_kind(self) -> ArgKind:
        return self.get_kind(PrintLevels.MID)

    @property
    def sub_arg(self) -> str:
        return self.get_arg(self.sub_kind)

    @property
    def unit_kind(self) -> ArgKind:
        return self.get_kind(PrintLevels.UNIT)

    @property
    def unit_arg(self) -> str:
        return self.get_arg(self.unit_kind)

    def get_arg_index_for(self, kind: ArgKind | PrintLevels) -> int:
        kind = self.get_kind(kind) if isinstance(kind, PrintLevels) else kind
        all_kind_args = getattr(self.context, kind)
        arg = self.get_arg(kind)
        return all_kind_args.index(arg)

    @cache  # noqa: B019
    def _is_first_in(self, kind: ArgKind) -> bool:
        match kind:
            case ArgKind.WORDS: all_kind_args = self.context.get_words_for(self.get_arg(ArgKind.FROM_LANGS))
            case _: all_kind_args = getattr(self.context, kind)
        arg = self.get_arg(kind)
        return not all_kind_args or not arg or arg == all_kind_args[0]

    def _is_rest_first_skip(self, skip_kind: ArgKind) -> bool:
        rest_kinds = [arg_kind for arg_kind in self.grouping if arg_kind != skip_kind]
        return all(
            self._is_first_in(other_kind) or not self.get_arg(other_kind)
            for other_kind in rest_kinds
        )

    def _is_all_sublevels_first(self, level: PrintLevels) -> bool:
        return all(self._is_first_in(self.get_kind(sublevel)) for sublevel in level.sublevels)

    def is_first_in_main_level(self) -> bool:
        return self.context.is_multi_from_langs() and self._is_all_sublevels_first(PrintLevels.MAIN)

    def is_first_in_mid_level(self) -> bool:
        is_multi_sub_group = getattr(self.context, f'is_multi_{self.sub_kind.value}')
        return is_multi_sub_group() and self._is_all_sublevels_first(PrintLevels.MID)

    def is_first_at_from_for(self, trans_kind: str) -> bool:
        return _.every([
            self.context.is_at_from(),
            getattr(self.context, trans_kind),
            self._is_rest_first_skip(ArgKind.FROM_LANGS),
        ])

    def is_first_at_from_inflection(self) -> bool:
        return self.is_first_at_from_for('inflection')

    def is_first_at_from_grammar(self) -> bool:
        return self.is_first_at_from_for('grammar')

    def is_first_at_from_overview(self) -> bool:
        return self.is_first_at_from_for('wiktio')

    def is_first_at_from_definition(self) -> bool:
        return self.is_first_at_from_for('definition')

    def is_translating(self) -> bool:
        return bool(self.get_arg(ArgKind.TO_LANGS))

    def is_first_at_to_for(self, outcome: str, main: Outcome) -> bool:
        return _.every([
            self.context.is_at_to(),
            getattr(self.context, outcome),
            main.is_success(),
        ])

    def is_first_at_to_inflection(self, main: Outcome) -> bool:
        return self.is_first_at_to_for('inflection', main)

    def is_first_at_to_grammar(self, main: Outcome) -> bool:
        return self.is_first_at_to_for('grammar', main)

    def is_first_at_to_overview(self, main: Outcome) -> bool:
        return self.is_first_at_to_for('wiktio', main)

    def is_first_at_to_definition(self, main: Outcome) -> bool:
        return self.is_first_at_to_for('definition', main)

    def get_header_label(self, level: PrintLevels) -> str:
        match self.get_kind(level):
            case ArgKind.WORDS: return '·'.join(self.context.get_word_group(self.get_arg_index_for(level)))
            case _: return self.get_arg(level)


@dataclass(frozen=False, init=False)
class Context:
    _conf: Conf = None

    words: tuple[str] = tuple()
    from_langs: tuple[str] = tuple()
    to_langs: tuple[str] = tuple()

    unmapped: tuple[bool] = tuple()

    at: str = UNSET
    wiktio: bool = UNSET
    inflection: bool = UNSET
    grammar: bool = UNSET
    definition: bool = UNSET
    pronunciation: bool = UNSET

    debug: bool = UNSET
    test: bool = UNSET

    assume: Assume = UNSET  # TODO: remove
    groupby: GroupBy | SpecialEnum = SpecialEnum.AUTO
    grouping_3: tuple[ArgKind, ArgKind, ArgKind] = UNSET
    grouping_2: tuple[ArgKind, ArgKind, ArgKind] = UNSET
    run_grouping: tuple[ArgKind] = UNSET
    indirect: Indirect = UNSET
    color: Box | Color = UNSET
    gather_data: GatherData = UNSET
    infervia: InferVia = UNSET
    retrain_on: RetrainOn = UNSET
    retrain: bool = UNSET

    loop: bool = UNSET

    mappings: Box | Mappings = UNSET

    _to_filter: ClassVar[set[str]] = {'args', 'reverse', 'add', 'delete', 'set', '_', 'orig_from_langs', 'orig_to_langs', 'langs'}

    def __init__(self, conf: Conf):
        self._conf: Conf = conf
        self.update(**conf.model_dump())

    def get_only_from_context(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as ar:
            if ar.args[0] != f"'{Context.__name__}' object has no attribute '{name}'":
                raise ar

    def __getattribute__(self, name: str) -> Any:
        try:
            if (val := super().__getattribute__(name)) is not UNSET and val != SpecialEnum.AUTO:
                return val
        except AttributeError as ar:
            if ar.args[0] != f"'{Context.__name__}' object has no attribute '{name}'":
                raise ar
        if (val := getattr(self._conf, name, UNSET)) is not UNSET and val != SpecialEnum.AUTO:
            return val
        if (val := getattr(Defaults, name, UNSET)) is not UNSET and val != SpecialEnum.AUTO:
            return val
        raise AttributeError(f'Attribute "{name}" not found ({val})')

    def update(self, **kwargs) -> None:
        kwargs = Box({key: val for key, val in kwargs.items() if key not in self._to_filter})
        if wrong_keys := {key for key in kwargs if not hasattr(self, key)}:
            raise ValueError(f'Context has no such keys: {wrong_keys}')
        for key, val in kwargs.items():
            if val is not UNSET:
                setattr(self, key, val)
        try:
            dict_attrs = _.pick_by(asdict(self), lambda val, key: _.is_dict(val) and not key.startswith('_'))
        except AttributeError as e:
            attribute = e.args[0].removeprefix('Attribute "').removesuffix('" not found')
            raise RuntimeError(f'Default not set for "{attribute}"')
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
    def grouping(self) -> list[ArgKind]:
        groups = self.grouping_2 if self.n_from_langs <= 1 else self.grouping_3
        if self.run_grouping:
            unuseds = [group for group in groups if group not in self.run_grouping]
            groups = list(self.run_grouping) + unuseds
        return list(groups)

    def get_words_for(self, from_lang: str) -> list[str]:
        i = self.from_langs.index(from_lang)
        return list(self.words[i::self.n_from_langs])

    def get_word_group(self, i: int) -> list[str]:
        return list(self.words[i:(i+1)*self.n_from_langs])

    @property
    def _from_lang_words_corr(self) -> dict[str, list[str]]:
        return {
            from_lang: self.get_words_for(from_lang)
            for from_lang in self.from_langs
        }

    def iterate_grouped_args(self) -> Iterable[ArgGroup]:
        from_lang_word_corr = self._from_lang_words_corr
        grouped_args = [getattr(self, kind) for kind in self.grouping]
        grouped_args = [args if args else [None] for args in grouped_args]
        for group in product(*grouped_args):
            arg_group = ArgGroup(group, grouping=self.grouping, context=self)
            from_lang, to_lang, word = arg_group.args
            if word not in from_lang_word_corr[from_lang]:
                continue
            yield arg_group

    @property
    def conf_langs(self) -> list[str]:
        return self._conf.langs

    @property
    def all_context_langs(self) -> list:
        return _.interleave(self.from_langs, self.to_langs)

    @property
    def n_from_langs(self) -> int:
        return len(self.from_langs)

    def is_multi_from_langs(self) -> bool:
        return self.n_from_langs > 1

    @property
    def n_to_langs(self) -> int:
        return len(self.to_langs)

    def is_multi_to_langs(self) -> bool:
        return self.n_to_langs > 1

    @property
    def n_words(self) -> int:
        return len(self.words)

    def is_multi_words(self) -> bool:
        return self.n_words > 1

    @property
    def unit_prefix_arg(self) -> str:
        unit_kind = self.grouping[-1]
        sub_kind = self.grouping[1]
        match getattr(self, f'is_multi_{unit_kind.value}')():
            case True: return unit_kind.value[:-1]
            case False: return sub_kind.value[:-1]

    @property
    def exit(self) -> bool:
        return not self.loop

    def is_at_from(self) -> bool:
        return self.at.startswith('f') or self.at.startswith('n')

    def is_at_to(self) -> bool:
        return self.at.startswith('t')

    @property
    def is_mappeds(self) -> list[bool]:
        return [o != w for o, w in zip(self.unmapped, self.words)]

    def is_mapped(self, word: str) -> bool:
        try:
            i = self.words.index(word)
        except ValueError:
            return False
        return self.is_mappeds[i]

    def get_unmmapped(self, word: str) -> str:
        i = self.words.index(word)
        return self.unmapped[i]
