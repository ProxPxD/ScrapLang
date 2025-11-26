from typing import Optional

from pandas import DataFrame

from src.lang_detecting.preprocessing.data import Columns as C


class SimpleDetector:
    def __init__(self, lang_script: DataFrame):
        self._lang_script = lang_script

    def _detect_on_by(self, column: str, values: set[str]) -> Optional[str]:
        fitting = self._lang_script[self._lang_script[column].apply(values.issubset)]
        if len(fitting) == 1:
            return fitting[C.LANG].iat[0]
        return None

    def detect_by_script(self, scripts: set[str]) -> Optional[str]:
        return self._detect_on_by(C.SCRIPTS, scripts)

    def detect_by_chars(self, chars: set[str]) -> Optional[str]:
        return self._detect_on_by(C.CHARS, chars)
