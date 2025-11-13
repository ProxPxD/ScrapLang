import warnings
from typing import Iterable, Sequence, Optional

import pytest
from _pytest.mark import ParameterSet, Mark
from pydash import chain as c
from toolz import unique

from testing.core.utils import apply

warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)


class TCG:
    """
    TCG - Test Case Generator
    """
    tcs = []

    @classmethod
    def generate_tcs(cls) -> list:
        return cls.tcs

    @classmethod
    def map(cls, tc) -> tuple:
        return tc

    @classmethod
    def map_to_many(cls, tc) -> Iterable:
        return [tc]

    @classmethod
    def gather_tag_before_mapping_to_many(cls, tc) -> Iterable[str]:
        return []

    @classmethod
    def gather_tags(cls, tc) -> Iterable[str]:
        return getattr(tc, 'tags', [])

    @classmethod
    def param_names(cls) -> Sequence[str] | str:
        raise ValueError()  # TODO: replace exception

    ###############
    # Undefinable #
    ###############

    @classmethod
    def _generate_marks(cls, tags: Iterable[str]) -> list[Mark]:
        return [cls._create_mark(tag) for tag in tags]

    @classmethod
    def _create_mark(cls, tag: str) -> Mark:
        name, *val = tag.split('/')
        kwargs = {} if not val else val[0]
        base_mark_decorator = pytest.mark.__getattr__(name.replace('-', '_'))
        return base_mark_decorator((), kwargs).mark

    @classmethod
    def _as_paramset(cls, tc, tags: list = None) -> ParameterSet:
        marks = cls._generate_marks(tags or [])
        tc = cls.map(tc)
        values = (tc, ) if hasattr(tc, '_asdict') or not isinstance(tc, tuple) else tc
        return pytest.param(*values, marks=marks)

    @classmethod
    @apply(list)
    def generate_params(cls) -> Iterable[ParameterSet]:
        for big_tc in cls.generate_tcs():
            big_tags = list(cls.gather_tag_before_mapping_to_many(big_tc))
            for lil_tc in cls.map_to_many(big_tc):
                lil_tags = list(cls.gather_tags(lil_tc))
                yield cls._as_paramset(lil_tc, unique(big_tags + lil_tags))

    @classmethod
    def create_name(cls, tc) -> Optional[str]:
        return None

    @classmethod
    def parametrize(cls, param_names=None, name_from: str | int | Sequence[str|int] = None, ids=None, **kwargs):
        get_tag_brackets = lambda tc: ' - (' + ', '.join(getattr(tc, 'tags', '')) + ') '
        if ids is None:
            name_from = (name_from, ) if isinstance(name_from, (str | int)) else name_from or ('name', 'short', 'descr')
            ids = lambda tc: cls.create_name(tc) or (c().at(*name_from).filter(bool).concat(tc).head()(tc) + get_tag_brackets(tc))
        param_names = param_names or cls.param_names()
        params = cls.generate_params()
        return pytest.mark.parametrize(param_names, params, ids=ids, **kwargs)
