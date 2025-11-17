from functools import reduce
from typing import Sequence, Optional

from pandas import DataFrame
import operator as op

from src.lang_detecting.lang_narrower import LangNarrower
from src.lang_detecting.mbidict import MBidict
from GlotScript import sp
from src.lang_detecting.preprocessing.data import Columns as C

from pydash import chain as c
import pydash as _

class LangPredictor:
    def __init__(self, lang_script: DataFrame):
        self._lang_script = lang_script
        # lang_script[lang_script[]]
        ...#self.lang_narrower = LangNarrower(lang_script)

    def predict_lang(self, words: Sequence[str]) -> Optional[str]:
        scripts = set(sp(''.join(words))[-1]['details'].keys())
        chars = reduce(op.or_, map(set, words))
        fitting_scripts = self._filter_by_scripts(self._lang_script, scripts)
        fitting_chars = self._filter_by_chars(fitting_scripts, chars)
        if len(fitting_chars) == 1:
            return fitting_chars[C.LANG].iat[0]
        # langs = self.lang_narrower.narrow_langs(scripts)
        # if len(langs) == 1:
        #     return  next(iter(langs))
        return None # fitting_chars

    @classmethod
    def _filter_by_scripts(cls, lang_script: DataFrame, scripts: set[str]) -> DataFrame:
        return lang_script[lang_script[C.SCRIPTS].apply(scripts.issubset)]

    @classmethod
    def _filter_by_chars(cls, lang_script: DataFrame, chars: set[str]) -> DataFrame:
        return lang_script[lang_script[C.CHARS].apply(chars.issubset)]
