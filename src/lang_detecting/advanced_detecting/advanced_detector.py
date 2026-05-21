import time
from typing import Any, Callable, Sequence

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

class Timer:
    def __init__(self) -> None:
        self._points = {}
        self._times = {}

    def time(self, label: Any = None, *, new_point: bool = False) -> None:
        point_label = None if None in self._points else label
        if point_label not in self._points:
            self._points[label] = time.time()
        else:
            self._times[label] = time.time() - self._points[point_label]
            self._points.pop(point_label)
        if new_point:
            self.time()

    def print_all(self) -> None:
        l_longest_label = max(len(label) for label in self._times)
        for label, t in self._times.items():
            print(f'{label}: {" " * (l_longest_label - len(label)) + str(t)}')

    def clear(self) -> None:
        self._points.clear()
        self._times.clear()


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame, valid_data_mgr: ValidDataMgr, conf: Conf):
        self.timer = Timer()
        loc = 'Adv'
        self.timer.time()
        self.model_io_mgr = ModelIOMgr()
        self.timer.time(f'{loc} Model IO')
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
        self.timer.time(f'{loc} K2C')
        self.tokenizer = MultiKindTokenizer(kinds_to_vocab, outputs, kind_to_specs=kind_to_specs)
        self.timer.time(f'{loc} Token')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timer.time(f'{loc} Device')
        self.moe = Moe(kinds_to_vocab, kinds_to_outputs, valmap(len, kind_to_specs), conf=self.conf).to(self.device)
        self.timer.time(f'{loc} Moe')
        self.timer.print_all()

    def retrain_model(self):
        dataset = BucketChunkDataset(self.valid_data_mgr.data, tokenizer=self.tokenizer, conf=self.conf, shuffle=True)
        optimizer = torch.optim.AdamW(self.moe.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        for epoch in range(self.conf.epochs):
            total_loss = 0.0
            n_records = 0
            for batch in dataset:
                kinds, words, specs, outputs = [t.to(self.device) for t in batch]
                n_records += (bs:=words.size(0))
                preds = self.moe(kinds, words, specs)
                loss = (preds - outputs).abs().sum()
                str_kind = self.tokenizer.detokenize_kind(kinds[0].item())
                str_word = ''.join(self.tokenizer.detokenize_input(words[0].tolist(), str_kind))
                pred_list = preds[0].tolist()
                p = pred_list.index(max(pred_list))
                p_lang = ''.join(self.tokenizer.detokenize_output(p))
                print(f'Word: {str_word}')
                print(f'Preds: {p_lang} : {pred_list}')
                print(f'Outputs: {outputs[0].tolist()}')
                print(f'Relative Loss: {loss/bs}\n')
                loss.backward()
                total_loss += loss.item()

                if n_records >= self.conf.accum_grad_bs:
                    optimizer.step()
                    optimizer.zero_grad()
                    n_records = 0
            if n_records:
                optimizer.step()
            print(f'Epoch {epoch + 1}/{self.conf.epochs}, Loss: {total_loss:.4f}')

