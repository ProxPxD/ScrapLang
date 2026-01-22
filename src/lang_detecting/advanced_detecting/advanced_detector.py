from collections import OrderedDict
from collections.abc import Collection
from typing import Sequence, Callable

import torch
from pandas import DataFrame
from pydash import chain as c

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.dataset import BucketChunkDataset
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import ModelIOMgr, KindToSpecialGroup, \
    TokenizedKindToGroupedVocab, KindToTokenMgr
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer, SpecGroup
from src.resouce_managing.valid_data import ValidDataMgr


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame, valid_data_mgr: ValidDataMgr, conf: Conf):
        self.model_io_mgr = ModelIOMgr()
        self.valid_data_mgr = valid_data_mgr
        self.conf = conf

        kinds_to_tokens_classes = self.model_io_mgr.extract_kinds_to_vocab_classes(lang_script)
        self.model_io_mgr.update_model_io_if_needed(kinds_to_tokens_classes)
        kinds_to_vocab, kinds_to_outputs = KindToTokenMgr.separate_kinds_tos(kinds_to_tokens_classes)
        outputs = c(kinds_to_outputs.values()).flatten().sorted_uniq().value()
        kind_to_specs: dict[str, Sequence[Callable]] = {
            'Latn': [str.isupper],
            'Cyrl': [str.isupper],
        }
        self.tokenizer = MultiKindTokenizer(kinds_to_vocab, outputs, kind_to_specs=kind_to_specs)
        self.moe = Moe(kinds_to_vocab, kinds_to_outputs, kind_to_specs, conf=self.conf).cuda()

    def retrain_model(self):
        sampler = BucketChunkDataset(self.valid_data_mgr.data, tokenizer=self.tokenizer, conf=self.conf)
        optimizer = torch.optim.AdamW(self.moe.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        for x in sampler:
            ...
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
