from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from src.context_domain import UNSET, ColorSchema, Mappings, assume, gather_data, groupby, indirect, infervia, retrain_on

ConfIndirect = Literal[*(indirect - {'conf'})]
ConfAssume = Literal[*(assume - {'conf'})]
ConfGatherData = Literal[*(gather_data - {'conf'})]
ConfInferVia = Literal[*(infervia - {'conf'})]
ConfGroupBy = Literal[*(groupby - {'conf'})]
ConfRetrainOn = Literal[*(retrain_on - {'conf'})]


class Conf(BaseModel):
    model_config = ConfigDict(extra='forbid')

    assume: ConfAssume = UNSET
    color: ColorSchema = UNSET  # TODO: test both dict and straight string colors
    groupby: ConfGroupBy = UNSET
    indirect: ConfIndirect = UNSET
    langs: list[str] = UNSET
    mappings: Mappings = UNSET
    gather_data: ConfGatherData = Field(default=UNSET, alias=AliasChoices('gather-data', 'gather_data'))
    infervia: ConfInferVia = UNSET
    retrain_on: ConfRetrainOn =  Field(default=UNSET, alias=AliasChoices('retrain-on', 'retrain_on', 'train-on', 'train_on'))
