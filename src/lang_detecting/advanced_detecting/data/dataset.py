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


class BucketChunkDataset(Dataset[list[int]]):
    def __init__(self,
            data: DataFrame,
            tokenizer: MultiKindTokenizer,
            conf: Conf,
            shuffle: bool = True,
            all_classes: list[str] = None,
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
        rng.seed(self.conf.seed or time.time())  # 9(3)
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
