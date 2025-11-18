from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pydash as _
from GlotScript import sp
from pandas import DataFrame

from src.constants import preinitialized
from src.resouce_managing.file import FileMgr
from src.resouce_managing.valid_data import Columns as GC

@preinitialized
@dataclass
class Columns:
    LANG: str = 'lang'
    CHARS: str = 'chars'
    SCRIPTS: str = 'scripts'

C = Columns


class DataProcessor:
    def __init__(self, *, valid_data_file: Path | str, lang_script_file: Path | str):
        self.valid_data_mgr = FileMgr(valid_data_file, create_if_not=True) if valid_data_file else None
        self.lang_script_mgr = FileMgr(lang_script_file, create_if_not=True) if lang_script_file else None

    @property
    def lang_script(self) -> DataFrame:
        return self.lang_script_mgr.content

    def _reanalyze(self, data: DataFrame) -> DataFrame:
        """
        :param data: [lang: str, word: str]
        :return:
        """
        lang_script = data.groupby(GC.LANG)[GC.WORD].apply(_.flow(''.join, set, sorted, ''.join)).reset_index()
        lang_script.rename(columns={GC.WORD: C.CHARS, GC.LANG: C.LANG}, inplace=True)
        lang_script[C.SCRIPTS] = lang_script[C.CHARS].apply(lambda w: set(sp(''.join(w))[-1]['details'].keys()))
        # langs_to_filter = []  # ['ja', 'zh']
        # print(lang_data[~lang_data[C.LANG].isin(langs_to_filter)].to_string(justify='left'))
        return lang_script

    def reanalyze(self) -> DataFrame:
        valid_data = self.valid_data_mgr.load()
        lang_script = self._reanalyze(valid_data)
        self.lang_script_mgr.save(lang_script)
        return self.lang_script
