import ast

import toolz
import torch
from pandas import DataFrame
import torch.nn as nn
from pydash import flow
from toolz import valmap
import pydash as _
from pydash import chain as c

from src.lang_detecting.preprocessing.data import LSC


class Router(nn.Module):
    ...

class ScriptExpert(nn.Module):
    def __init__(self, n_classes: int, vocab_size: int):
        super().__init__()
        ...

class Combiner(nn.Module):
    ...


class Moe(nn.Module):
    def __init__(self, kinds_to_classes, emb_dim=32):
        super().__init__()
        self.emb_dim = emb_dim
        # self.kind_model = nn.ModuleDict(valmap(flow(len, ScriptExpert), kinds_to_tokens_and_classes))
        # self.kind_model = nn.ModuleDict({kindfor kind, classes in kinds_to_tokens_and_classes.items()}))

    # def forward(self, words) -> torch.Tensor:
    #     return self.net(x)
