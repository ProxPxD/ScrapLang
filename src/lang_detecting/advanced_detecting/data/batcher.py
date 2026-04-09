from __future__ import annotations

from functools import cached_property
from typing import Optional, TYPE_CHECKING

import pydash as _
import torch
from pandas import DataFrame
from pydash import chain as c
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import Cols, TensorBatch
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer, Tokens
from src.resouce_managing.valid_data import VDC

if TYPE_CHECKING:
    from src.lang_detecting.advanced_detecting.conf import Chunking, Conf


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
        chunk_buckets = data.groupby(data[Cols.LEN].apply(lambda l: (l+self.chunking.overlap)//self.chunking.size+1 if l > self.chunking.size else 1))  # noqa: E741
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
        tokenized_tokens = self.pad_seq(tokens, pad_val=self.pad_token)
        tokenized_spec_groups = self.pad_seq(specs, pad_val=-1).to_sparse()
        t_targets, m_taget_kind = self._create_targets(targets, tokenized_kinds)
        return tokenized_kinds, tokenized_tokens, tokenized_spec_groups, t_targets, m_taget_kind

    def _create_targets(self, targets: list, tokenized_kinds: Tensor) -> tuple[Tensor, Tensor]:
        tokenized_targets = _.map_(targets, c().map(self.tokenizer.tokenize_target_plural))
        t_targets = self._one_hot(tokenized_targets)
        l_kind_targets = _.map_(tokenized_kinds.tolist(), self.tokenizer.get_tokenized_targets_for_kind)
        m_taget_kind = self._one_hot(l_kind_targets)
        return t_targets, m_taget_kind

    def _one_hot(self, tensor: list, dim: int = None) -> Tensor:
        dim = dim or self.tokenizer.n_target_tokens
        one_hot = torch.zeros(len(tensor), dim, dtype=torch.long)
        for j, targets in enumerate(tensor):
            one_hot[j, targets] = 1
        return one_hot

    @classmethod
    def pad_seq(cls, to_tensors: list, pad_val: int = 0) -> Tensor:
        tensorized = pad_sequence(_.map_(to_tensors, torch.tensor), batch_first=True, padding_value=pad_val).int()
        return tensorized

