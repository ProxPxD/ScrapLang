from abc import ABC
from enum import StrEnum, Enum
from typing import Any, Iterable, Literal, Sequence

from pydantic import RootModel
from pydash import chain as c
import pydash as _


UNSET = object()

class Unsupported(Exception):
    def __init__(self, arg: Any):
        msg: str = ''
        match arg:
            case Enum() as enum: msg = f'Unsupported {type(enum)} value: {enum.value}!'
            case _: msg = f'Unsupported {type(arg)} value: {arg}'
        super().__init__(msg)

class IVals(Enum):
    @classmethod
    @property
    def choices(cls) -> list[str]:
        return [member.value for member in cls]


class SpecialEnum(StrEnum, IVals):
    CONF = 'conf'  # TODO: Replace all with "auto"
    AUTO = 'auto'

class IWithAuto(IVals):
    @classmethod
    @property
    def choices_plus(cls) -> list[str]:
        return cls.choices + SpecialEnum.choices

class GroupBy(StrEnum, IWithAuto):
    LANG = 'lang'
    WORD = 'word'

    @classmethod
    @property
    def name(cls) -> str:
        return 'groupby'


indirect = {'on', 'off', 'fail', 'conf'}
assume = {'lang', 'word', 'no'}
gather_data = {'all', 'ai', 'time', 'off', 'conf'}
infervia = {'all', 'ai', 'time', 'last', 'off', 'conf'}
retrain_on = {'gather', 'flag'}
at = {'from', 'to', 'f', 't', 'none'}


Indirect = Literal[*indirect]
Assume = Literal[*assume]
GatherData = Literal[*gather_data]
InferVia = Literal[*infervia]
RetrainOn = Literal[*retrain_on]
Mappings = dict[str, list[dict[str, str]] | dict[str, str]]
At = Literal[*at]

color_names = {'main', 'pronunciation'}
ColorNames = Literal[*color_names]
ColorFormat = str | Sequence[int]
Color = dict[ColorNames, ColorFormat] | str

class ColorSchema(RootModel[Color]):
    pass

