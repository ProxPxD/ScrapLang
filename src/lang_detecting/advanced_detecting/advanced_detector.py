from pandas import DataFrame

from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import ModelIOMgr
from src.lang_detecting.advanced_detecting.tokenizer import MultiTokenizer


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame):
        self.model_io_mgr = ModelIOMgr()

        kinds_to_tokens_classes = self.model_io_mgr.extract_kinds_to_tokens_classes(lang_script)
        self.model_io_mgr.update_model_io_if_needed(kinds_to_tokens_classes)
        kinds_to_tokens, kinds_to_classes = self.model_io_mgr.separate_kinds_tos(kinds_to_tokens_classes)

        self.tokenizer = MultiTokenizer(kinds_to_tokens)
        self.moe = Moe(kinds_to_classes)

