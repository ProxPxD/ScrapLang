from pandas import DataFrame

import torch

from src.lang_detecting.advanced_detecting.model import Moe


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame):
        self.moe = Moe(lang_script)
