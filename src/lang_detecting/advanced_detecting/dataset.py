from __future__ import annotations

import random
import time
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from functools import cached_property

import pydash as _
import torch
from GlotScript import sp
from pandas import DataFrame
from pydash import chain as c, flow
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer
from src.resouce_managing.valid_data import VDC


@dataclass(frozen=True)
class Cols:
    LEN = 'len'
    KIND = 'kind'
    TOKENS = 'tokens'
    SPECS = 'specs'
    DECODE = 'decode'
    N_UNIQ = 'n_uniq'

TensorBatch = tuple[Tensor, Tensor, Tensor, Tensor]

random.seed(time.time())

class BucketChunkDataset(Dataset[list[int]]):
    def __init__(self,
            data: DataFrame,
            tokenizer: MultiKindTokenizer,
            conf: Conf,
            shuffle: bool = True,
            all_classes: list[str] = None
    ):
        """
        :param valid_data_mgr:
        :param max_batch_size: None means all
        :param shuffle:
        """
        super().__init__()
        self.conf: Conf = conf
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        data = self._process_data(data)
        self.class_counts = OrderedDict(data.explode(VDC.LANG)[VDC.LANG].value_counts())
        self._all_classes = all_classes or list(self.class_counts.keys())
        self.class_weights = self._compute_weights(self._all_classes, self.class_counts)
        self.batches = self._map_data_to_tensor_batches(data)

    def _compute_weights(self, all_classes: list[str], class_counts: OrderedDict[str, int], bias: float = None) -> Tensor:
        bias = bias or self.conf.freq_bias
        present_classes = class_counts.keys()
        counts = torch.tensor([class_counts[c] for c in present_classes])
        freq = counts / counts.sum()
        raw_weights = freq ** -bias
        raw_weights /= raw_weights.mean()

        present_idxs = torch.tensor(_.map_(present_classes, all_classes.index), dtype=torch.long)
        weights = torch.ones(len(all_classes), dtype=raw_weights.dtype)
        weights[present_idxs] = raw_weights
        return weights

    def _process_data(self, data: DataFrame) -> DataFrame:
        data = self._prefilter(data)
        data = self._form_model_form(data)
        data = self._expand_rows(data)
        data = self._filter_bad_train_data(data)
        data = data.groupby([Cols.KIND, Cols.TOKENS, Cols.SPECS], sort=False).agg({
            VDC.WORD: flow(set, list, ''.join),  # Should be same as c().get(0), but this allows to detect discrepancies
            VDC.LANG: flow(set, sorted, list),
        }).reset_index()
        #data = data.sort_values([Cols.LEN, Cols.N_UNIQ])
        pass
        # TODO: make it work with multikind langs like japanese
        #data = data[data.apply(lambda row: all(char in self._kinds_to_shared[row[Cols.KIND]] for char in row[VDC.WORD]), axis=1)]
        return data

    def _prefilter(self, data: DataFrame) -> DataFrame:
        data = data[~data[VDC.IS_MAPPED]][[VDC.LANG, VDC.WORD]]
        return data

    def _form_model_form(self, data: DataFrame) -> DataFrame:
        data = copy(data)
        data[Cols.KIND] = data[VDC.WORD].apply(lambda w: next(iter(sp(w)[-1]['details'].keys())))
        data[Cols.TOKENS] = data.apply(lambda row: self.tokenizer.tokenize_input(row[VDC.WORD], row[Cols.KIND]), axis=1)
        data[Cols.SPECS] = data.apply(lambda row: self.tokenizer.tokenize_spec_groups(row[VDC.WORD], row[Cols.KIND]), axis=1)
        return data

    def _expand_rows(self, data: DataFrame) -> DataFrame:
        data = copy(data)
        data[Cols.DECODE] = data.apply(lambda row: self.tokenizer.detokenize_input(row[Cols.TOKENS], row[Cols.KIND]), axis=1)
        data[Cols.LEN] = data[Cols.TOKENS].str.len()
        data[Cols.N_UNIQ] = data[Cols.DECODE].apply(c().filter(self.tokenizer.unknown.__contains__).apply(len))
        return data

    def _filter_bad_train_data(self, data: DataFrame) -> DataFrame:
        m_uniq = data[Cols.N_UNIQ] > 0
        m_enough_non_uniq = data[Cols.LEN] - data[Cols.N_UNIQ] >= 5
        m_right_ratio = (data[Cols.LEN] - 3) / data[Cols.N_UNIQ] >= 2
        m_long = data[Cols.LEN] >= self.conf.data.input_len_thresh
        data = data[m_long & (~m_uniq | m_enough_non_uniq & m_right_ratio)]
        class_counts = data[VDC.LANG].value_counts()
        numerous_enough = class_counts[class_counts >= self.conf.data.record_count_thresh].index
        data = data[data[VDC.LANG].isin(numerous_enough)]
        return data

    def _map_data_to_tensor_batches(self, data: DataFrame) -> list[TensorBatch]:
        batches: list[TensorBatch] = []
        len_bucketed = data.sort_values(Cols.LEN, ascending=False).groupby(Cols.LEN, sort=False)
        for word_length, bucket in len_bucketed:
            batch_size = self.conf.max_batch_size or len(bucket)
            for i in range(0, len(bucket), batch_size):
                batch_data = list(bucket.iloc[i: i + batch_size][[Cols.KIND, VDC.WORD, VDC.LANG]].itertuples(index=False, name=None))
                batch = self._create_batch(batch_data)
                batches.append(batch)
        return batches

    def _create_batch(self, batch: list[tuple]) -> TensorBatch:
        kinds, words, outputs = tuple(zip(*batch))
        tokenized_kinds = Tensor(_.map_(kinds, self.tokenizer.tokenize_kind)).int()
        tokenized_words = Tensor([self.tokenizer.tokenize_input(word, kind) for word, kind in zip(words, kinds)]).int()
        tokenized_spec_groups = [Tensor(self.tokenizer.tokenize_spec_groups(word, kind)) for word, kind in zip(words, kinds)]
        tokenized_spec_groups = pad_sequence(tokenized_spec_groups, batch_first=True, padding_value=0).int().to_sparse()
        tokenized_outputs = _.map_(outputs, c().map(self.tokenizer.tokenize_target))
        one_hot_encoded_outputs = torch.zeros(len(tokenized_outputs), self.tokenizer.n_target_tokens, dtype=torch.float32)
        for j, outputs in enumerate(tokenized_outputs):
            one_hot_encoded_outputs[j, outputs] = 1 / len(outputs)
        return tokenized_kinds, tokenized_words, tokenized_spec_groups, one_hot_encoded_outputs

    def __iter__(self):
        if self.shuffle:
            rng = random.Random()
            rng.seed(9 or time.time())  # 9(3)
            rng.shuffle(self.batches)
        for batch in self.batches:
            kinds, words, specs, outputs = batch
            yield kinds, words, specs, outputs

    def __len__(self) -> int:
        return len(self.batches)

    @cached_property
    def batch_sizes(self) -> list[int]:
        return [batch[0].shape[0] for batch in self.batches]

    @cached_property
    def n_records(self) -> int:
        return sum(self.batch_sizes)
