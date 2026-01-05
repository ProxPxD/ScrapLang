from pandas import DataFrame

from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import ModelIOMgr, KindToGroupedVocab
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame):
        self.model_io_mgr = ModelIOMgr()

        kinds_to_tokens_classes = self.model_io_mgr.extract_kinds_to_vocab_classes(lang_script)
        self.model_io_mgr.update_model_io_if_needed(kinds_to_tokens_classes)
        kinds_to_vocab, kinds_to_classes = self.model_io_mgr.separate_kinds_tos(kinds_to_tokens_classes)
        kinds_to_grouped_vocab: KindToGroupedVocab = self.model_io_mgr.group_up_kinds_to_vocab(kinds_to_vocab)

        self.tokenizer = MultiKindTokenizer(kinds_to_grouped_vocab)
        self.moe = Moe(kinds_to_classes)

