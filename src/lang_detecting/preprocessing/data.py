from functools import cache

from pandas import DataFrame
from pandas._config import display

from src.constants import Paths
from src.lang_detecting.mbidict import mbidict, MBidict
from src.resouce_managing.file import FileMgr
import pydash as _
from GlotScript import sp


class DataPreprocessor:
    def __init__(self):
        self._valid_data_file_mgr = FileMgr(Paths.VALID_DATA_FILE)

    @property
    @cache
    def data(self) -> DataFrame:
        return self._valid_data_file_mgr.load()

    def process(self, data: DataFrame):
        """
        :param data: [lang: str, word: str]
        :return:
        """
        lang_data = data.groupby('lang')['word'].apply(_.flow(''.join, set, sorted, ''.join)).reset_index()
        lang_data.rename(columns={'word': 'chars'}, inplace=True)
        lang_data['script'] = lang_data['chars'].apply(lambda w: set(sp(''.join(w))[-1]['details'].keys()))
        langs_to_filter = []  # ['ja', 'zh']
        print(lang_data[~lang_data['lang'].isin(langs_to_filter)].to_string(justify='left'))
        return lang_data
