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
            include_special: bool = True,
    ):
        super().__init__()
        self.conf: Conf = conf
        self.shuffle = shuffle
        self.include_special = include_special
        self.tokenizer = tokenizer

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
