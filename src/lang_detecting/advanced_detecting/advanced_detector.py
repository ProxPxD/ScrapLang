import logging
from itertools import combinations
from typing import Any, Collection

from pandas import DataFrame
from pydash import flow, spread, partial
from torchgen.utils import OrderedSet

from src.constants import Paths
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.preprocessing.data import LSC
import pydash as _
from src.resouce_managing.file import FileMgr
import operator as op
from pydash import chain as c


class AdvancedDetector:
    def __init__(self, lang_script: DataFrame):
        model_io = FileMgr(Paths.MODEL_IO_FILE, create_if_not=True)
        old_script_langs = model_io.load()
        shary_script = self.extract_scripts_to_any_shared_chars_for_langs(lang_script)
        if old_script_langs != shary_script:
            logging.debug(f'Updating model IO')
            model_io.save(shary_script) # TODO: Uncomment after retraining functionality
            # raise ValueError('Model requires retraining and thats unsupported and unhandled for now')
        self.moe = Moe(shary_script)

    @classmethod
    def filter_any_shared_chars(cls, chars_group: Collection[str]):
        combineds = c(chars_group).apply(lambda cs: combinations(map(set, cs), 2))
        any_shareds = combineds.map(spread(op.and_))
        all_any_shareds = any_shareds.reduce(lambda a, b: a | b, set())
        as_strs = all_any_shareds.to_list().sort().join()
        return as_strs.value()

    @classmethod
    def extract_scripts_to_any_shared_chars_for_langs(cls, lang_script: DataFrame) -> dict[str, dict[str, str | list]]:
        script_lang = lang_script.explode(LSC.SCRIPTS).rename(columns={LSC.SCRIPTS: LSC.SCRIPT})
        sclc = script_common_langs_chars = (script_lang.groupby(LSC.SCRIPT, as_index=False).agg({
            LSC.LANG: flow(sorted, list),
            LSC.CHARS: cls.filter_any_shared_chars
        }).rename(columns={LSC.LANG: LSC.LANGS}))
        sclc = sclc[(sclc[LSC.LANGS].map(len) > 1) & (sclc[LSC.CHARS].map(len) > 0)].reset_index(drop=True)
        shary_script = {row[LSC.SCRIPT]: {LSC.LANGS: row[LSC.LANGS], LSC.CHARS: ''.join(row[LSC.CHARS])} for idx, row in sclc.iterrows()}
        return shary_script
