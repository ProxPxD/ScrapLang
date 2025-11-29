import logging
from itertools import combinations
from typing import Any

from pandas import DataFrame
from pydash import flow, spread
from torchgen.utils import OrderedSet

from src.constants import Paths
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.preprocessing.data import C
import pydash as _
from src.resouce_managing.file import FileMgr
import operator as op
from pydash import chain as c


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame):
        model_io = FileMgr(Paths.MODEL_IO_FILE)
        old_script_langs = model_io.load()
        shary_script = self.extract_shary_scripts(lang_script)
        if old_script_langs != shary_script:
            logging.debug(f'Updating model IO')
            # model_io.save(shary_script) # TODO: Uncomment after retraining functionality
            raise ValueError('Model requires retraining and thats unsupported and unhandled for now')
        self.moe = Moe(shary_script)

    @classmethod
    def extract_shary_scripts(cls, lang_script: DataFrame) -> dict[str, dict[str, str | list]]:
        script_lang = lang_script.explode(C.SCRIPTS).rename(columns={C.SCRIPTS: C.SCRIPT})
        sclc = script_common_langs_chars = script_lang.groupby(C.SCRIPT, as_index=False).agg({
            C.LANG: flow(sorted, list),
            C.CHARS: c().apply(lambda cs: combinations(map(set, cs), 2)).map(spread(op.and_)).reduce_(lambda a, b: a|b, set()).to_list().sort().join()
        }).rename(columns={C.LANG: C.LANGS})
        sclc = sclc[(sclc[C.LANGS].map(len) > 1) & (sclc[C.CHARS].map(len) > 0)].reset_index(drop=True)
        shary_script = {row[C.SCRIPT]: {C.LANGS: row[C.LANGS], C.CHARS: ''.join(row[C.CHARS])} for idx, row in sclc.iterrows()}
        return shary_script
