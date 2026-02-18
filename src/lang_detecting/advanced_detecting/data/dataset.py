from __future__ import annotations

import random
import time
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from itertools import zip_longest

import pandas as pd
import pydash as _
import torch
from GlotScript import sp
from pandas import DataFrame
from pydash import chain as c, flow
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer, Tokens
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
            all_classes: list[str] = None,
            augment: bool = False,
            include_special: bool = True,
    ):
        super().__init__()
        self.conf: Conf = conf
        self.shuffle = shuffle
        self.include_special = include_special
        self.tokenizer = tokenizer
        data = self._process_data(data)
        self.class_counts = OrderedDict(data.explode(VDC.LANG)[VDC.LANG].value_counts())
        self._all_classes = all_classes or list(self.class_counts.keys())
        self.class_weights = self._compute_weights(self._all_classes, self.class_counts)
        self.batches = self._map_data_to_tensor_batches(data)

    @property
    def augment(self) -> bool:
        return self.conf.data.augment.is_augmenting

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
        data = self._filter_extend(data)
        if self.augment:
            data = self._augment_data(data)
        data = data.groupby([Cols.KIND, Cols.TOKENS, Cols.SPECS], sort=False).agg({
            VDC.WORD: flow(list, c().get(0)),
            VDC.LANG: flow(set, sorted, list),
            Cols.LEN: flow(list, c().get(0)),
        }).reset_index()
        return data

    def _filter_extend(self, data: DataFrame) -> DataFrame:
        data = self._remove_mappings(data)
        data = self._form_model_form(data)
        data = self._expand_rows(data)
        data = self._filter_out_bad_train_data(data)
        return data

    @classmethod
    def _remove_mappings(cls, data: DataFrame) -> DataFrame:
        mask = ~data[VDC.IS_MAPPED] if VDC.IS_MAPPED in data.columns else pd.Series(True, index=data.index)
        data = data[mask][[VDC.LANG, VDC.WORD]]
        return data.drop_duplicates()

    def _form_model_form(self, data: DataFrame) -> DataFrame:
        data = copy(data)
        data[VDC.WORD] = data[VDC.WORD].apply(lambda w: w[1:-1] if Tokens.BOS in w else tuple(w))
        def get_kind(word: str):
            details: dict = sp(''.join(word))[-1]['details']
            return next(iter(details.keys())) if details else None
        data[Cols.KIND] = data[VDC.WORD].apply(get_kind)
        data = data[~data[Cols.KIND].isna()]
        data[Cols.TOKENS] = data.apply(lambda row: self.tokenizer.tokenize_input([Tokens.BOS, *list(row[VDC.WORD]), Tokens.BOS], row[Cols.KIND]), axis=1)
        data[Cols.SPECS] = data.apply(lambda row: self.tokenizer.tokenize_spec_groups(['|', *list(row[VDC.WORD]), '|'], row[Cols.KIND]), axis=1)  # A bit silly fix, but let it slide
        return data.drop_duplicates()

    def _expand_rows(self, data: DataFrame) -> DataFrame:
        data = copy(data)
        data[Cols.DECODE] = data.apply(lambda row: self.tokenizer.detokenize_input(row[Cols.TOKENS], row[Cols.KIND]), axis=1)
        data[Cols.LEN] = data[Cols.TOKENS].str.len()
        data[Cols.N_UNIQ] = data[Cols.DECODE].apply(c().filter(self.tokenizer.unknown.__contains__).apply(len))
        return data

    def _filter_out_bad_train_data(self, data: DataFrame) -> DataFrame:
        m_uniq = data[Cols.N_UNIQ] > 0
        m_enough_non_uniq = data[Cols.LEN] - data[Cols.N_UNIQ] >= self.conf.data.input_non_uniq_enough_count
        m_right_ratio = (data[Cols.LEN] - 5) / data[Cols.N_UNIQ] >= self.conf.data.input_right_ratio
        m_long = data[Cols.LEN] >= self.conf.data.input_len_thresh
        data = data[m_long & (~m_uniq | m_enough_non_uniq & m_right_ratio)]
        class_counts = data[VDC.LANG].value_counts()
        numerous_enough = class_counts[class_counts >= self.conf.data.record_count_thresh].index
        data = data[data[VDC.LANG].isin(numerous_enough)]
        return data

    def _augment_data(self, data: DataFrame) -> DataFrame:
        data = self._augment_unknown(data)
        data = self._filter_extend(data)  # TODO: Think of more rigorous ratio threshold for the augmented data (mark them temporally with a column)
        return data

    def _augment_unknown(self, data: DataFrame) -> DataFrame:
        pre_patterns, post_patterns = self._extract_patterns(data)
        def apply_pattern(word: list[str], pattern: list[str], reverse: bool = False):
            if reverse:
                word, pattern = word[::-1], pattern[::-1]
            applied = [u if u and l not in {Tokens.BOS} else l for l, u in zip_longest(word, pattern)]
            if reverse:
                applied = applied[::-1]
            return tuple(applied)

        dfs = [data]
        for patterns, reverse in [(pre_patterns, False), (post_patterns, True)]:
            for pattern in patterns:
                pre_df = copy(data)
                pre_df[VDC.WORD] = pre_df[Cols.DECODE].apply(lambda word: apply_pattern(word, pattern, reverse=reverse))
                dfs.append(pre_df)
        return pd.concat(dfs).drop_duplicates([VDC.WORD, VDC.LANG])

    def _extract_patterns(self, data) -> tuple[list[list[str]], list[list[str]]]:
        unknown_patterns = (
            c(data[Cols.DECODE].to_list())
            .map(c().map_(flow(self.tokenizer.unknown.__eq__, int)))
            .uniq()
        )
        norm = c().uniq().filter(any).map(c().map(lambda t: self.tokenizer.unknown if t else ''))
        pre_patterns = unknown_patterns.commit().map(lambda p: p[:self.conf.data.augment.pre_augment_size]).value()
        post_patterns = unknown_patterns.map(lambda p: p[-self.conf.data.augment.post_augment_size:]).value()
        return norm(pre_patterns), norm(post_patterns)

    def _map_data_to_tensor_batches(self, data: DataFrame) -> list[TensorBatch]:
        batches: list[TensorBatch] = []
        len_bucketed = data.sort_values(Cols.LEN, ascending=False).groupby(Cols.LEN, sort=False)
        for word_length, bucket in len_bucketed:
            batch_size = self.conf.max_batch_size or len(bucket)
            for i in range(0, len(bucket), batch_size):
                batch_data = list(bucket.iloc[i: i + batch_size][[Cols.KIND, Cols.TOKENS, Cols.SPECS, VDC.LANG]].itertuples(index=False, name=None))
                batch = self._create_batch(batch_data)
                batches.append(batch)
        return batches

    def _create_batch(self, batch: list[tuple]) -> TensorBatch:
        kinds, tokens, specs, outputs = tuple(zip(*batch))
        tokenized_kinds = Tensor(_.map_(kinds, self.tokenizer.tokenize_kind)).int()
        tokenized_words = Tensor(tokens).int()
        tokenized_spec_groups = pad_sequence(_.map_(specs, torch.tensor), batch_first=True, padding_value=0).int().to_sparse()
        tokenized_outputs = _.map_(outputs, c().map(self.tokenizer.tokenize_target))
        one_hot_encoded_targets = torch.zeros(len(tokenized_outputs), self.tokenizer.n_target_tokens, dtype=torch.long)
        for j, outputs in enumerate(tokenized_outputs):
            one_hot_encoded_targets[j, outputs] = 1 #/ len(outputs)
        return tokenized_kinds, tokenized_words, tokenized_spec_groups, one_hot_encoded_targets

    def shuffle_batches(self) -> None:
        rng = random.Random()
        rng.seed(9 or time.time())  # 9(3)
        rng.shuffle(self.batches)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_batches()
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
