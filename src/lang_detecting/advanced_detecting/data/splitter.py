from collections import Counter

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from src.lang_detecting.advanced_detecting.conf import Conf
from src.resouce_managing.valid_data import VDC


class Splitter:
    def __init__(self, conf: Conf):
        self.conf = conf

    @property
    def min_n_label(self) -> int:
        return self.conf.data.valset.min_n_label

    def split(self, df: DataFrame):
        df = df.sample(frac=1, random_state=self.conf.seed).reset_index(drop=True)
        n_val_records = int(len(df) * self.conf.data.valset.size)
        val_indices = self._get_min_val_indices(df)
        n_rest_val = n_val_records - len(val_indices)
        val_indices.update(df.drop(val_indices).sample(n=n_rest_val, random_state=self.conf.seed).index)
        train_df = df.drop(val_indices).reset_index(drop=True)
        val_df = df.loc[list(val_indices)].reset_index(drop=True)
        return train_df, val_df

    def _get_min_val_indices(self, df: DataFrame) -> set[pd.Index]:
        used_labels = set(df[VDC.LANG].explode())
        val_indices, label_counter = set(), Counter()
        label_counter.clear()
        for idx, labels in df[VDC.LANG].items():
            if all(label_counter[l] >= self.min_n_label for l in labels):
                continue

            val_indices.add(idx)
            for l in labels:
                label_counter[l] += 1

            if all(label_counter[l] >= self.min_n_label for l in used_labels):
                break
        return val_indices

