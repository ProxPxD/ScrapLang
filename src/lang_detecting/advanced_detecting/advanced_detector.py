from pandas import DataFrame

from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.preprocessing.data import C


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame):
        script_lang = lang_script.explode(C.SCRIPTS).rename(columns={C.SCRIPTS: C.SCRIPT})
        script_lang = (
            script_lang.groupby(C.SCRIPT)[C.LANG]
            .apply(list)
            .reset_index(name=C.LANGS)
            .sort_values(C.SCRIPT)
            .reset_index(drop=True)
        )
        ambiguous_script_langs = script_lang[script_lang[C.LANGS].map(len) > 1]
        self.moe = Moe(ambiguous_script_langs)
