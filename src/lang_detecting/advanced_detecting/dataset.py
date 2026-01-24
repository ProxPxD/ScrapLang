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
from torch.nn.utils.rnn import pad_sequence
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
        self.batches: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        data = data[~data[VDC.IS_MAPPED]][[VDC.LANG, VDC.WORD]].drop_duplicates()
        data = data.groupby(VDC.WORD, sort=False).agg({VDC.LANG: flow(set, sorted, list)}).reset_index()
        data[LEN:='len'] = data[VDC.WORD].str.len()
        # TODO: make it work with multikind langs like japanese
        data[KIND:='kind'] = data[VDC.WORD].apply(lambda w: next(iter(sp(w)[-1]['details'].keys())))
        len_bucketed = data.sort_values(LEN, ascending=False).groupby(LEN, sort=False)
        for length, bucket in len_bucketed:
            batch_size = conf.max_batch_size or len(bucket)
            for i in range(0, len(bucket), batch_size):
                batch = list(bucket.iloc[i: i + batch_size][[KIND, VDC.WORD, VDC.LANG]].itertuples(index=False, name=None))
                kinds, words, outputs = tuple(zip(*batch))
                tokenized_kinds = Tensor(_.map_(kinds, tokenizer.tokenize_kind)).int()
                tokenized_words = Tensor([tokenizer.tokenize_input(word, kind) for word, kind in zip(words, kinds)]).int()
                tokenized_spec_groups = [Tensor(tokenizer.tokenize_spec_groups(word, kind)) for word, kind in zip(words, kinds)]
                tokenized_spec_groups = pad_sequence(tokenized_spec_groups, batch_first=True, padding_value=0).int()
                tokenized_outputs = _.map_(outputs, c().map(tokenizer.tokenize_output))
                one_hot_encoded_outputs = torch.zeros(len(tokenized_outputs), tokenizer.n_output_tokens, dtype=torch.float32)
                for j, outputs in enumerate(tokenized_outputs):
                    one_hot_encoded_outputs[j, outputs] = 1 / len(outputs)
                self.batches.append((tokenized_kinds, tokenized_words, tokenized_spec_groups, one_hot_encoded_outputs))

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
