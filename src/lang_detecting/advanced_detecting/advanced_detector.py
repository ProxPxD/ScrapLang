from pandas import DataFrame

import torch




class AdvancedDetector:
    def __init__(self, lang_script: DataFrame):
        self.lang_script = lang_script
