from dataclasses import dataclass

import pydash as _
from GlotScript import sp
from pandas import DataFrame

from src.constants import preinitialized
from src.resouce_managing.valid_data import Columns as GC

@preinitialized
@dataclass
class Columns:
    LANG: str = 'lang'
    CHARS: str = 'chars'
    SCRIPTS: str = 'scripts'

C = Columns


class DataPreprocessor:
    def process(self, data: DataFrame):
        """
        :param data: [lang: str, word: str]
        :return:
        """
        lang_data = data.groupby(GC.LANG)[GC.WORD].apply(_.flow(''.join, set, sorted, ''.join)).reset_index()
        lang_data.rename(columns={GC.WORD: C.CHARS, GC.LANG: C.LANG}, inplace=True)
        lang_data[C.SCRIPTS] = lang_data[C.CHARS].apply(lambda w: set(sp(''.join(w))[-1]['details'].keys()))
        langs_to_filter = []  # ['ja', 'zh']
        print(lang_data[~lang_data[C.LANG].isin(langs_to_filter)].to_string(justify='left'))
        return lang_data
