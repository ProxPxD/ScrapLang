from abc import ABC, abstractmethod
from typing import Callable, Protocol

import pydash as _
from pandas import DataFrame
from pydash import flow

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer


class Resources:
    def __init__(self, tokenizer: MultiKindTokenizer = None, conf: Conf = None):
        self.tokenizer: MultiKindTokenizer = tokenizer
        self.conf: Conf = conf


class Step(Protocol):
    def perform(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, *args, **kwargs) -> DataFrame:
        ...


class AbstractStep(ABC):
    def __init__(self, resources: Resources = None, precond: Callable[[DataFrame], bool] = None, **kwargs):
        self.resources = resources
        self.precond = precond or _.constant(True)

    @abstractmethod
    def perform(self, data: DataFrame) -> DataFrame:
        pass

    def __call__(self, *args, **kwargs):
        return self.perform(*args, **kwargs)


class SimpleStep(AbstractStep):
    def __init__(self, func: Callable[[DataFrame], DataFrame], **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def perform(self, data: DataFrame) -> DataFrame:
        return self.func(data)


class SeqStep(AbstractStep):
    def __init__(self, *steps: Step, **kwargs):
        self.steps = steps
        super().__init__(**kwargs)

    def perform(self, data: DataFrame) -> DataFrame:
        if not self.precond(data):
            return data
        return flow(*self.steps)(data)
