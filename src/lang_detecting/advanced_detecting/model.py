import ast

import torch
from pandas import DataFrame
import torch.nn as nn

from src.lang_detecting.preprocessing.data import C


class Router(nn.Module):
    ...

class ScriptExpert(nn.Module):
    ...

class Combiner(nn.Module):
    ...


class Moe(nn.Module):
    def __init__(self, script_lang: DataFrame):
        super().__init__()
        # script_lang[]
        pass
