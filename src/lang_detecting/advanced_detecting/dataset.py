import random
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

import pydash as _
import torch
import torch.nn.functional as F
from GlotScript import sp
from pandas import DataFrame
from pydash import chain as c
from pydash import flow
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

class BucketChunkDataset(Dataset[list[int]]):
    def __init__(self,
            data: DataFrame,
            tokenizer: MultiKindTokenizer,
            conf: Conf,
            shuffle: bool = True,
            kinds_to_shared: dict[str, str] = None
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
        self._kinds_to_shared: dict[str, str] = kinds_to_shared or {}
        data = self._process_data(data)
        self.class_counts = OrderedDict(data.explode(VDC.LANG).groupby(VDC.LANG, sort=True).agg({VDC.WORD: len}).reset_index().values)
        self.batches = self._map_data_to_tensor_batches(data)

    @property
    def class_weights(self):
        t_class_count = torch.tensor(list(self.class_counts.values()), dtype=torch.float)
        b = self.conf.weights_beta
        weights = (1 - b) / (1 - b ** t_class_count)
        weights = weights / weights.mean()
        return weights

    def _process_data(self, data: DataFrame) -> DataFrame:
        data = self._prefilter(data)
        data = self._form_grouped_model_form(data)
        data = self._expand_rows(data)
        data = self._filter_bad_train_data(data)
        data = data.sort_values([Cols.LEN, Cols.N_UNIQ])
        pass
        # TODO: make it work with multikind langs like japanese
        #data = data[data.apply(lambda row: all(char in self._kinds_to_shared[row[Cols.KIND]] for char in row[VDC.WORD]), axis=1)]
        return data

    def _prefilter(self, data: DataFrame) -> DataFrame:
        data = data[~data[VDC.IS_MAPPED]][[VDC.LANG, VDC.WORD]]
        return data

    def _form_grouped_model_form(self, data: DataFrame) -> DataFrame:
        data[Cols.KIND] = data[VDC.WORD].apply(lambda w: next(iter(sp(w)[-1]['details'].keys())))
        data[Cols.TOKENS] = data.apply(lambda row: self.tokenizer.tokenize_input(row[VDC.WORD], row[Cols.KIND]), axis=1)
        data[Cols.SPECS] = data.apply(lambda row: self.tokenizer.tokenize_spec_groups(row[VDC.WORD], row[Cols.KIND]), axis=1)
        data = data.groupby([Cols.KIND, Cols.TOKENS, Cols.SPECS], sort=False).agg({
            VDC.WORD: flow(set, list, ''.join),  # Should be same as c().get(0), but this allows to detect discrepancies
            VDC.LANG: flow(set, sorted, list),
        }).reset_index()
        return data

    def _expand_rows(self, data: DataFrame) -> DataFrame:
        data[Cols.DECODE] = data.apply(lambda row: self.tokenizer.detokenize_input(row[Cols.TOKENS], row[Cols.KIND]), axis=1)
        data[Cols.LEN] = data[Cols.TOKENS].str.len()
        data[Cols.N_UNIQ] = data[Cols.DECODE].apply(c().filter(self.tokenizer.unknown.__contains__).apply(len))
        return data

    def _filter_bad_train_data(self, data: DataFrame) -> DataFrame:
        m_uniq = data[Cols.N_UNIQ] > 0
        m_enough_non_uniq = data[Cols.LEN] - data[Cols.N_UNIQ] >= 5
        m_right_ratio = (data[Cols.LEN] - 3) / data[Cols.N_UNIQ] >= 2
        m_long = data[Cols.LEN] >= self.conf.data.len_thresh
        data = data[m_long & (~m_uniq | m_enough_non_uniq & m_right_ratio)]
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
            random.shuffle(self.batches)
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
