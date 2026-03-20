from __future__ import annotations

import math
from functools import cached_property
from itertools import repeat
from typing import Any

import pydash as _
import torch
from pandas import DataFrame
from pydash import chain as c
from toolz import curry
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.lang_detecting.advanced_detecting.conf import Chunking, Conf
from src.lang_detecting.advanced_detecting.data.preprocessing import PreprocessorFactory
from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import Cols, TensorBatch
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer, Tokens
from src.resouce_managing.valid_data import VDC
import torch.nn.functional as F


class Batcher:
    def __init__(self, conf: Conf, tokenizer: MultiKindTokenizer):
        self.conf: Conf = conf
        self.tokenizer: MultiKindTokenizer = tokenizer

    @property
    def chunking(self) -> Chunking:
        return self.conf.data.chunking

    @cached_property
    def pad_token(self) -> int:
        return self.tokenizer.tokenize_common(Tokens.PAD)

    def batch_data_up(self, data: DataFrame) -> list[TensorBatch]:
        batches: list[TensorBatch] = []
        data = data.sort_values(Cols.LEN, ascending=False)
        chunk_buckets = data.groupby(data[Cols.LEN].apply(lambda l: (l+self.chunking.overlap)//self.chunking.size+1 if l > self.chunking.size else 1))
        for n_chunks, bucket in chunk_buckets:
            n_chunks: int
            batch_size = self.conf.train.max_batch_size or len(bucket)
            for i in range(0, len(bucket), batch_size):
                batch_data = list(bucket.iloc[i: i + batch_size][[Cols.KIND, Cols.TOKENS, Cols.SPECS, VDC.LANG]].itertuples(index=False, name=None))
                batch = self._create_batch(batch_data)
                batches.append(batch)
        return batches

    def _create_batch(self, batch: list[tuple]) -> TensorBatch:
        kinds, tokens, specs, targets = tuple(zip(*batch))
        tokenized_kinds = torch.tensor(_.map_(kinds, self.tokenizer.tokenize_kind)).int()
        tokenized_tokens = self.pad_seq(tokens, dim=-1, pad_val=self.pad_token)
        tokenized_spec_groups = self.pad_seq(specs, dim=-2, pad_val=-1).to_sparse()
        tokenized_outputs = _.map_(targets, c().map(self.tokenizer.tokenize_target))
        one_hot_encoded_targets = torch.zeros(len(tokenized_outputs), self.tokenizer.n_target_tokens, dtype=torch.long)
        for j, targets in enumerate(tokenized_outputs):
            one_hot_encoded_targets[j, targets] = 1
        return tokenized_kinds, tokenized_tokens, tokenized_spec_groups, one_hot_encoded_targets

    def pad_seq(self, to_tensors: list, dim: int, pad_val: int = 0) -> Tensor:
        tensorized = pad_sequence(_.map_(to_tensors, torch.tensor), batch_first=True, padding_value=pad_val).int()
        # s_unpadded = tensorized.shape[dim]
        # n_chunks = math.ceil(s_unpadded / 16)
        # n_to_pad = (self.chunking.size * n_chunks - s_unpadded)
        # from_end = -(dim if dim < 0 else dim - len(tensorized.shape))
        # initial_zeroes = (0, 0) * (from_end - 1)
        # tensorized = F.pad(tensorized, (*initial_zeroes, 0, n_to_pad), value=pad_val)
        return tensorized

