from pandas import DataFrame
from pydash import chain as c, flow

from src.lang_detecting.advanced_detecting.data.consts import Cols
from src.lang_detecting.advanced_detecting.data.step import AbstractStep
from src.resouce_managing.valid_data import VDC


class Grouper(AbstractStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def perform(self, data: DataFrame) -> DataFrame:
        data = data.groupby([Cols.KIND, Cols.TOKENS, Cols.SPECS], sort=False).agg({
            VDC.WORD: flow(list, c().get(0)),
            VDC.LANG: flow(set, sorted, list),
            Cols.LEN: flow(list, c().get(0)),
        }).reset_index()
        return data
