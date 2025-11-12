from typing import Literal, Sequence

from pydantic import RootModel

UNSET = object()

indirect = {'on', 'off', 'fail', 'conf'}
assume = {'lang', 'word', 'no'}
gather_data = {'all', 'ai', 'time', 'off', 'conf'}
infervia = {'all', 'ai', 'time', 'last', 'off', 'conf'}
groupby = {'lang', 'word', 'conf'}
at = {'from', 'to', 'f', 't', 'none'}


Indirect = Literal[*indirect]
Assume = Literal[*assume]
GatherData = Literal[*gather_data]
InferVia = Literal[*infervia]
GroupBy = Literal[*groupby]
Mappings = dict[str, list[dict[str, str]] | dict[str, str]]
At = Literal[*at]

color_names = {'main', 'pronunciation'}
ColorNames = Literal[*color_names]
ColorFormat = str | Sequence[int]
Color = dict[ColorNames, ColorFormat] | str

class ColorSchema(RootModel[Color]):
    pass

