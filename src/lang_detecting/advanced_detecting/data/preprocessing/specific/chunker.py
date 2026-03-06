from pandas import DataFrame

from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import Cols
from src.lang_detecting.advanced_detecting.data.preprocessing.core.step import AbstractStep
from src.resouce_managing.valid_data import VDC


class Chunker(AbstractStep):
    def __init__(self, size: int, to_col: str, from_col, **kwargs):
        super().__init__(**kwargs)
        self.size: int = size
        self.to_col = to_col
        self.from_col = from_col

    def perform(self, df: DataFrame) -> DataFrame:
        df[self.to_col] = df[self.from_col].apply(lambda w: [w[i:i + self.size] for i in range(len(w) - self.size + 1)])
        df = df.explode(self.to_col).drop_duplicates()
        df = df[~df[self.to_col].isna()]
        return df
