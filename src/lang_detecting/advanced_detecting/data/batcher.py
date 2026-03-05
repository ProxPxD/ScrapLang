from __future__ import annotations

import random
import time
from collections import OrderedDict
from functools import cached_property

import pydash as _
import torch
from pandas import DataFrame
from pydash import chain as c
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import Cols, TensorBatch
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer
from src.resouce_managing.valid_data import VDC
from src.lang_detecting.advanced_detecting.conf import Conf


class Batcher:
    def __init__(self, conf: Conf, tokenizer: MultiKindTokenizer):
        self.conf: Conf = conf
        self.tokenizer: MultiKindTokenizer = tokenizer

    def batch_data_up(self, data: DataFrame) -> list[TensorBatch]:
        batches: list[TensorBatch] = []
        len_bucketed = data.sort_values(Cols.LEN, ascending=False).groupby(Cols.LEN, sort=False)
        for word_length, bucket in len_bucketed:
            batch_size = self.conf.train.max_batch_size or len(bucket)
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
            one_hot_encoded_targets[j, outputs] = 1  # / len(outputs)
        return tokenized_kinds, tokenized_words, tokenized_spec_groups, one_hot_encoded_targets

