import ast
from dataclasses import dataclass
from pathlib import Path

from GlotScript import sp
from pandas import DataFrame
from pydash import chain as c
from pydash import flow

from src.constants import preinitialized
from src.resouce_managing.file import FileMgr
from src.resouce_managing.valid_data import VDC, ValidDataMgr


@preinitialized
@dataclass
class LangScriptColumns:
    LANG: str = 'lang'
    LANGS: str = 'langs'
    CHARS: str = 'chars'
    SCRIPTS: str = 'scripts'
    SCRIPT: str = 'script'

LSC = LangScriptColumns

def adjust_lang_script(lang_script: DataFrame) -> DataFrame:
    if lang_script is not None:
        lang_script[LSC.SCRIPTS] = lang_script[LSC.SCRIPTS].apply(ast.literal_eval)
    return lang_script


class DataProcessor:
    def __init__(self, *, valid_data_mgr: ValidDataMgr, lang_script_file: Path | str):
        self.valid_data_mgr = valid_data_mgr
        self.lang_script_mgr = FileMgr(lang_script_file, create_if_not=True, func=adjust_lang_script) if lang_script_file else None

    @property
    def lang_script(self) -> DataFrame:
        return self.lang_script_mgr.content


    def _generate_script_summary(self, data: DataFrame) -> DataFrame:
        """
        :param data: [lang: str, word: str]
        :return:
        """
        lang_script = data.groupby(VDC.LANG)[VDC.WORD].apply(flow(''.join, set, c().flat_map(lambda c: [c, c.upper()]), set, sorted, ''.join)).reset_index()
        lang_script.rename(columns={VDC.WORD: LSC.CHARS, VDC.LANG: LSC.LANG}, inplace=True)
        lang_script[LSC.SCRIPTS] = lang_script[LSC.CHARS].apply(lambda w: set(sp(''.join(w))[-1]['details'].keys()))
        return lang_script

    def generate_script_summary(self) -> DataFrame:
        valid_data = self.valid_data_mgr.load()
        lang_script = self._generate_script_summary(valid_data)
        self.lang_script_mgr.save(lang_script)
        return self.lang_script
