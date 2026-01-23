from typing import Callable, Sequence

import torch
from pandas import DataFrame
from pydash import chain as c
from toolz import valmap

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.dataset import BucketChunkDataset
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import KindToTokenMgr, ModelIOMgr
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer
from src.resouce_managing.valid_data import ValidDataMgr

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.moe = Moe(kinds_to_vocab, kinds_to_outputs, valmap(len, kind_to_specs), conf=self.conf).to(self.device)

    def retrain_model(self):
        dataset = BucketChunkDataset(self.valid_data_mgr.data, tokenizer=self.tokenizer, conf=self.conf)
        optimizer = torch.optim.AdamW(self.moe.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        # for batch in dataset:
        #     kinds, words, specs, outputs = batch
        #     preds = self.moe(kinds, words, specs)
        #     loss = (preds - outputs).abs().sum()
        #     loss.backward()
        b = 4
        while True:
            B = 2 ** b
            try:
                L = 32
                print(f'Testing batch size = 2^{b:<2} = {B:<7} on length = {L}')
                dummy_kinds = torch.randint(0, 2, (B, )).to(self.device)
                dummy_words = torch.randint(0, 10, (B, L,)).to(self.device)
                dummy_specs = torch.randint(0, 2,  (B, L, 1)).to(self.device)
                out = self.moe(dummy_kinds, dummy_words, dummy_specs)
                loss = out.sum()
                loss.backward()
                b += 1
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e, '\n', f'  for B = 2^{b} = {B}')
                    break
                else:
                    raise e
