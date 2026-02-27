from __future__ import annotations

from copy import copy
from typing import Callable

from pandas import DataFrame

from src.lang_detecting.advanced_detecting.data.preprocessing.core.step import AbstractStep


class RowTransform(AbstractStep):
    def __init__(self,
            func: Callable, *,
            post_func: Callable[[DataFrame], DataFrame] = None,
            from_col: str = None, to_col: str = None, col: str = None,
            **kwargs
        ):
        self.from_col = from_col or col
        self.to_col = to_col or col
        self.func = func
        self.post_func = post_func
        super().__init__(**kwargs)

    def perform(self, data: DataFrame) -> DataFrame:
        data = copy(data)
        transformed = data[self.from_col].apply(self.func) if self.from_col else data.apply(self.func, axis=1)
        if self.to_col:
            data[self.to_col] = transformed
        else:
            data = transformed
        if self.post_func:
            data = self.post_func(data)
        return data

