from pandas import DataFrame

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.dataset import BucketSampler
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import ModelIOMgr, KindToGroupedVocab
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer
from src.resouce_managing.valid_data import ValidDataMgr

import torch

class AdvancedDetector:
    def __init__(self, lang_script: DataFrame, valid_data_mgr: ValidDataMgr, conf: Conf):
        self.model_io_mgr = ModelIOMgr()
        self.valid_data_mgr = valid_data_mgr
        self.conf = conf

        kinds_to_tokens_classes = self.model_io_mgr.extract_kinds_to_vocab_classes(lang_script)
        self.model_io_mgr.update_model_io_if_needed(kinds_to_tokens_classes)
        kinds_to_vocab, kinds_to_classes = self.model_io_mgr.separate_kinds_tos(kinds_to_tokens_classes)
        kinds_to_grouped_vocab: KindToGroupedVocab = self.model_io_mgr.group_up_kinds_to_vocab(kinds_to_vocab)

        self.tokenizer = MultiKindTokenizer(kinds_to_grouped_vocab, )
        self.moe = Moe(kinds_to_grouped_vocab, kinds_to_classes, conf=self.conf).cuda()

    def retrain_model(self):
        sampler = BucketSampler(self.valid_data_mgr, tokenizer=self.tokenizer, max_batch_size=self.conf.max_batch_size)
        optimizer = torch.optim.AdamW(self.moe.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)

        B = 1
        while True:
            try:
                dummy_words = torch.randint(0, 10, (B, 7, 10)).cuda()
                dummy_scripts = torch.randint(0, 1, (B, )).cuda()
                out = self.moe(dummy_words, dummy_scripts)
                loss = out.sum()
                loss.backward()
                B *= 2
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    break
