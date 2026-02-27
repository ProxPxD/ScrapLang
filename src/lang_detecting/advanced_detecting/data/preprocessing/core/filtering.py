from __future__ import annotations

from typing import Callable

import pydash as _
from pandas import DataFrame
from pydash import chain as c, flow

from src.lang_detecting.advanced_detecting.data.preprocessing.core.step import AbstractStep


class ColFilter(AbstractStep):
    def __init__(self, func: Callable = None, col: str = None, output_cols: list[str] = None, precond: Callable[[DataFrame], bool] = c().identity(True), **kwargs):
        self.precond = precond
        self.func = func
        self.col = col
        self.output_cols = output_cols
        super().__init__(**kwargs)

    def perform(self, data: DataFrame) -> DataFrame:
        if not self.precond(data):
            return data
        match self.col:
            case None: pivot = data
            case _ if self.col in data.columns: pivot = data[self.col]
            case _: pivot = None
        # noinspection PyUnboundLocalVariable
        if self.func and pivot is not None:  # noinspection PyUnboundLocalVariable
            mask = self.func(pivot)
            data = data[mask]
        if self.output_cols:
            data = data[self.output_cols].drop_duplicates()
        return data

    @property
    def _decol_func(self) -> Callable:
        return self.func if not self.col else lambda df: self.func(df[self.col])

    def __and__(self, other) -> ColFilter:
        return ColFilter(func=flow(self, other))

    def __or__(self, other) -> ColFilter:
        return ColFilter(func=lambda df: self._decol_func(df) | other._decol_func(df))

    def __invert__(self):
        return ColFilter(func=lambda df: ~self._decol_func(df))
