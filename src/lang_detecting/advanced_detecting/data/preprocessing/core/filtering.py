from __future__ import annotations

from typing import Any, Callable, Self

from pandas import DataFrame
from pydash import chain as c
from pydash import flow

from src.lang_detecting.advanced_detecting.data.preprocessing.core.step import AbstractStep


class ColFilter(AbstractStep):
    def __init__(self, mask_func: Callable = None, col: str = None, output_cols: list[str] = None, precond: Callable[[DataFrame], bool] = c().identity(True), **kwargs: Any):  # noqa: FBT003
        self.precond = precond
        self.func = mask_func
        self.col = col
        self.output_cols = output_cols
        super().__init__(**kwargs)

    def perform(self, df: DataFrame) -> DataFrame:
        if not self.precond(df):
            return df
        match self.col:
            case None: pivot = df
            case _ if self.col in df.columns: pivot = df[self.col]
            case _: pivot = None
        # noinspection PyUnboundLocalVariable
        if self.func and pivot is not None:  # noinspection PyUnboundLocalVariable
            mask = self.func(pivot)
            df = df[mask]
        if self.output_cols:
            df = df[self.output_cols].drop_duplicates()
        return df

    @property
    def _decol_func(self) -> Callable:
        return self.func if not self.col else lambda df: self.func(df[self.col])

    def __and__(self, other: Self) -> Self:
        return ColFilter(mask_func=flow(self, other))

    def __or__(self, other: Self) -> Self:
        return ColFilter(mask_func=lambda df: self._decol_func(df) | other._decol_func(df))  # noqa: SLF001

    def __invert__(self) -> Self:
        return ColFilter(mask_func=lambda df: ~self._decol_func(df))
