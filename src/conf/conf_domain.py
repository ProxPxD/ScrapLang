from enum import Enum
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from src.context_domain import ColorSchema, GroupBy, PrintLevels, Mappings, SpecialEnum, UNSET, assume, gather_data, indirect, infervia, retrain_on

ConfIndirect = Literal[*(indirect - {'conf'})]
ConfAssume = Literal[*(assume - {'conf'})]
ConfGatherData = Literal[*(gather_data - {'conf'})]
ConfInferVia = Literal[*(infervia - {'conf'})]
ConfRetrainOn = Literal[*(retrain_on - {'conf'})]


class Conf(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        use_enum_values=False,
        json_encoders = {Enum: lambda e: e.value},
    )

    assume: ConfAssume = UNSET
    color: ColorSchema = UNSET  # TODO: test both dict and straight string colors
    groupby: GroupBy | SpecialEnum = UNSET
    grouping_2: list[PrintLevels] = UNSET
    grouping_3: list[PrintLevels] = UNSET
    indirect: ConfIndirect = UNSET
    langs: list[str] = UNSET
    mappings: Mappings = UNSET
    gather_data: ConfGatherData = Field(default=UNSET, alias=AliasChoices('gather-data', 'gather_data'))
    infervia: ConfInferVia = UNSET
    retrain_on: ConfRetrainOn =  Field(default=UNSET, alias=AliasChoices('retrain-on', 'retrain_on', 'train-on', 'train_on'))
