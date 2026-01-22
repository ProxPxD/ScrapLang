import random
from functools import cached_property

import pydash as _
import torch
import torch.nn.functional as F
from GlotScript import sp
from pandas import DataFrame
from pydash import chain as c
from pydash import flow
from torch import Tensor
from torch.utils.data import Dataset

from src.conf import Conf
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer
from src.resouce_managing.valid_data import VDC


class BucketChunkDataset(Dataset[list[int]]):
    def __init__(self, data: DataFrame, tokenizer: MultiKindTokenizer, conf: Conf, shuffle: bool = True):
        """
        :param valid_data_mgr:
        :param max_batch_size: None means all
        :param shuffle:
        """
        super().__init__()
        self.conf = conf
        self.shuffle = shuffle
        self.batches: list[tuple[Tensor, Tensor, Tensor]] = []
        data = data[~data[VDC.IS_MAPPED]][[VDC.LANG, VDC.WORD]].drop_duplicates()
        data = data.groupby(VDC.WORD, sort=False).agg({VDC.LANG: flow(set, sorted, list)}).reset_index()
        data[LEN:='len'] = data[VDC.WORD].str.len()
        # TODO: make it work with multikind langs like japanese
        data[KIND:='kind'] = data[VDC.WORD].apply(lambda w: next(iter(sp(w)[-1]['details'].keys())))
        len_bucketed = data.sort_values(LEN, ascending=False).groupby(LEN, sort=False)
        for length, bucket in len_bucketed:
            batch_size = conf.max_batch_size or len(bucket)
            for i in range(0, len(bucket), batch_size):
                batch = list(bucket.iloc[i: i + batch_size][[VDC.WORD, KIND, VDC.LANG]].itertuples(index=False, name=None))
                words, kinds, outputs = tuple(zip(*batch))
                tokenized_words = Tensor([tokenizer.tokenize_input(word, kind) for word, kind in zip(words, kinds)])
                tokenized_kinds = Tensor(_.map_(kinds, tokenizer.tokenize_kind)).int()
                tokenized_outputs = _.map_(outputs, c().map(tokenizer.tokenize_output))
                tokenized_spec_groups = Tensor([tokenizer.tokenize_spec_groups(word, kind) for word, kind in zip(words, kinds)])
                one_hot_encoded_outputs = torch.zeros(len(tokenized_outputs), tokenizer.n_outputs, dtype=torch.int32)
                for j, outputs in enumerate(tokenized_outputs):
                    one_hot_encoded_outputs[j, outputs] = 1
                self.batches.append((tokenized_words, tokenized_kinds, one_hot_encoded_outputs))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            words, kinds, output = batch

    def __len__(self) -> int:
        return len(self.batches)

    @cached_property
    def batch_sizes(self) -> list[int]:
        return [batch[0].shape[0] for batch in self.batches]

    @cached_property
    def n_records(self) -> int:
        return sum(self.batch_sizes)
