import random
from collections import defaultdict
from functools import cached_property

import torch
from pydash import flow
from torch.utils.data import Dataset, Sampler
from pydash import chain as c

from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer
from src.resouce_managing.valid_data import ValidDataMgr, VDC


class BucketSampler(Sampler[list[int]]):
    def __init__(self, valid_data_mgr: ValidDataMgr, tokenizer: MultiKindTokenizer, max_batch_size: int = None, shuffle: bool = True):
        """
        :param valid_data_mgr:
        :param max_batch_size: None means all
        :param shuffle:
        """
        super().__init__()
        self.tokenizer =  tokenizer
        self.shuffle = shuffle
        self.batches = []
        data = valid_data_mgr.data
        data = data[~data[VDC.IS_MAPPED]][[VDC.LANG, VDC.WORD]].drop_duplicates()
        data = data.groupby(VDC.WORD, sort=False).agg({VDC.LANG: flow(set, sorted, list)}).reset_index()
        data[LEN:='len'] = data[VDC.WORD].str.len()
        len_bucketed = data.sort_values(LEN, ascending=False).groupby(LEN, sort=False)
        for length, bucket in len_bucketed:
            batch_size = max_batch_size or len(bucket)
            for i in range(0, len(bucket), batch_size):
                self.batches.append(list(bucket.iloc[i: i + batch_size][[VDC.WORD, VDC.LANG]].itertuples(index=False, name=None)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            pass
            yield batch

    def __len__(self) -> int:
        return len(self.batches)

    @cached_property
    def n_records(self) -> int:
        return sum(map(len, self.batches))
