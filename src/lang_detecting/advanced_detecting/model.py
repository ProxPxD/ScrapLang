import ast

import toolz
import torch
from pandas import DataFrame
import torch.nn as nn
from pydash import flow
from toolz import valmap
import pydash as _
from pydash import chain as c

from src.lang_detecting.preprocessing.data import C


class Router(nn.Module):
    ...

class ScriptExpert(nn.Module):
    def __init__(self, n_classes: int, vocab_size: int):
        super().__init__()
        ...

class Combiner(nn.Module):
    ...


class Moe(nn.Module):
    def __init__(self, model_kinds_to_classes):
        super().__init__()
        # self.kind_model = nn.ModuleDict(valmap(flow(len, ScriptExpert), model_kinds_to_classes))
        # self.kind_model = nn.ModuleDict({kindfor kind, classes in model_kinds_to_classes.items()}))
