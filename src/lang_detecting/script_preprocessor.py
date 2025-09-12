from functools import cache

from pandas import DataFrame
from pandas._config import display

from src.constants import Paths
from src.lang_detecting.mbidict import mbidict, MBidict
from src.resouce_managing.file import FileMgr
import pydash as _


class ScriptPreprocessor:
    def __init__(self):
        self._valid_data_file_mgr = FileMgr(Paths.VALID_DATA_FILE)

    @property
    @cache
    def data(self) -> DataFrame:
        return self._valid_data_file_mgr.load()

    def process(self):
        ...

    def create_lang_script_correspondence(self) -> MBidict:
        # lang_script = mbidict()
        lang_script_df = self.data.groupby('lang')['word'].apply(_.flow(''.join, set, sorted, list)).reset_index()
        # lang_script_df.style.set_properties(**{'text-align': 'left'})

        print(lang_script_df[~lang_script_df['lang'].isin(['ja', 'zh'])].to_string(justify='left'))
