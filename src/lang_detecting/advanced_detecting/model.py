import ast

import torch
from pandas import DataFrame
import torch.nn as nn

from src.lang_detecting.preprocessing.data import C


class Moe(nn.Module):
    def __init__(self, lang_script: DataFrame):
        super().__init__()
        script_lang = lang_script.explode(C.SCRIPTS).rename(columns={C.SCRIPTS: C.SCRIPT})
        script_lang = (
            script_lang.groupby(C.SCRIPT)[C.LANG]
            .apply(list)
            .reset_index(name=C.LANGS)
            .sort_values(C.SCRIPT)
            .reset_index(drop=True)
        )
        pass
