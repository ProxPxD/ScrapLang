from functools import cache

from pandas import DataFrame
from pandas._config import display

from src.constants import Paths
from src.lang_detecting.mbidict import mbidict, MBidict
from src.resouce_managing.file import FileMgr
import pydash as _
from GlotScript import sp


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
        lang_script = mbidict('langs', 'scripts')
        lang_script_df = self.data.groupby('lang')['word'].apply(_.flow(''.join, set, sorted, ''.join)).reset_index()
        for row in lang_script_df.itertuples():
            pred = sp(row.word)
            lang_script.left[row.lang].extend(scripts := list(pred[-1]['details'].keys()))

        print(lang_script)
        print(lang_script.pretty_string)

        langs_to_filter = []# ['ja', 'zh']
        print(lang_script_df[~lang_script_df['lang'].isin(langs_to_filter)].to_string(justify='left'))
