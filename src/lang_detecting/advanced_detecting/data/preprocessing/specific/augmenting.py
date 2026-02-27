from copy import copy
from itertools import zip_longest

import pandas as pd
from pandas import DataFrame
from pydash import chain as c, flow

from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import Cols
from src.lang_detecting.advanced_detecting.data.preprocessing.core.step import AbstractStep
from src.lang_detecting.advanced_detecting.tokenizer import Tokens
from src.resouce_managing.valid_data import VDC


class Augmenter(AbstractStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def perform(self, data: DataFrame) -> DataFrame:
        # TODO: Think of more rigorous ratio threshold for the augmented data (mark them temporally with a column)
        pre_patterns, post_patterns = self._extract_patterns(data)
        dfs = [data]
        for patterns, reverse in [(pre_patterns, False), (post_patterns, True)]:
            for pattern in patterns:
                pre_df = copy(data)
                pre_df[VDC.WORD] = pre_df[Cols.DECODE].apply(lambda word: self.apply_pattern(word, pattern, reverse=reverse))
                dfs.append(pre_df)
        return pd.concat(dfs).drop_duplicates([VDC.WORD, VDC.LANG])

    def _extract_patterns(self, data) -> tuple[list[list[str]], list[list[str]]]:
        tokenizer = self.resources.tokenizer
        conf = self.resources.conf
        unknown_patterns = (
            c(data[Cols.DECODE].to_list())
            .map(c().map_(flow(tokenizer.unknown.__eq__, int)))
            .uniq()
        )
        norm = c().uniq().filter(any).map(c().map(lambda t: tokenizer.unknown if t else ''))
        pre_patterns = unknown_patterns.commit().map(lambda p: p[conf.data.augment.pre_augment_size]).value()
        post_patterns = unknown_patterns.map(lambda p: p[conf.data.augment.post_augment_size:]).value()
        return norm(pre_patterns), norm(post_patterns)

    @classmethod
    def apply_pattern(cls, word: list[str], pattern: list[str], reverse: bool = False):
        if reverse:
            word, pattern = word[::-1], pattern[::-1]
        applied = [u if u and l not in {Tokens.BOS} else l for l, u in zip_longest(word, pattern)]
        if reverse:
            applied = applied[::-1]
        return tuple(applied)
