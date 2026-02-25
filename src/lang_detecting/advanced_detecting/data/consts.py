from dataclasses import dataclass

from torch import Tensor

TensorBatch = tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class Cols:
    LEN = 'len'
    KIND = 'kind'
    TOKENS = 'tokens'
    SPECS = 'specs'
    DECODE = 'decode'
    N_UNIQ = 'n_uniq'
