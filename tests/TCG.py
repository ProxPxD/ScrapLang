
from typing import Iterable, Sequence

import pytest
from _pytest.mark import MarkDecorator, ParameterSet
from pydash import chain as c
from toolz import unique
from pydash import flow


def apply(*map_funcs):
    def decorator(f):
        def wrapper(*args, **kwarg):
            return flow(*map_funcs)(f(*args, **kwarg))
        return wrapper
    return decorator


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
    def map_to_many(cls, tc) -> Iterable | list:
        return [tc]

    @classmethod
    def gather_tag_before_mapping_to_many(cls, tc) -> Iterable[str] | Sequence[str]:
        return []

    @classmethod
    def gather_tags(cls, tc) -> Iterable[str] | Sequence[str]:
        return []

    @classmethod
    def param_names(cls) -> list[str] | str:
        raise ValueError()  # TODO: replace exception

    ###############
    # Undefinable #
    ###############

    @classmethod
    def _generate_marks(cls, tags: Iterable[str] | Sequence[str]) -> list[MarkDecorator]:
        return [getattr(pytest.mark, tag) for tag in tags]

    @classmethod
    def _as_paramset(cls, tc, tags: list = None) -> ParameterSet:
        marks = cls._generate_marks(tags or [])
        tc = cls.map(tc)
        values = (tc, ) if hasattr(tc, '_asdict') or not isinstance(tc, tuple) else tc
        return pytest.param(*values, marks=marks)

    @classmethod
    @apply(list)
    def generate_params(cls) -> list[ParameterSet]:
        for big_tc in cls.generate_tcs():
            big_tags = list(cls.gather_tag_before_mapping_to_many(big_tc))
            for lil_tc in cls.map_to_many(big_tc):
                lil_tags = list(cls.gather_tags(lil_tc))
                yield cls._as_paramset(lil_tc, unique(big_tags + lil_tags))

    @classmethod
    def parametrize(cls, param_names=None, name_from: str | int | Sequence[str|int] = None, ids=None, **kwargs):
        if ids is None:
            name_from = (name_from, ) if isinstance(name_from, (str | int)) else name_from or ('name', 'short', 'descr')
            ids = lambda tc: c().at(*name_from).filter(bool).concat(tc).head()(tc)
        param_names = param_names or cls.param_names()
        params = cls.generate_params()
        return pytest.mark.parametrize(param_names, params, ids=ids, **kwargs)
