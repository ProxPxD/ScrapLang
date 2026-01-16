import ast
import operator as op
from functools import reduce
from typing import Sequence, Optional

from GlotScript import sp
from pandas import DataFrame

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.preprocessing.data import LSC
from src.lang_detecting.simple_detecting import SimpleDetector
from src.resouce_managing.valid_data import ValidDataMgr

try:
    import torch
    HAS_LIB_TORCH = True
except ImportError:
    HAS_LIB_TORCH = False

if HAS_LIB_TORCH:
    from src.lang_detecting.advanced_detecting import AdvancedDetector


class Detector:
    def __init__(self, lang_script: DataFrame, valid_data_mgr: ValidDataMgr):
        self.lang_script = lang_script
        self.simple_detector = SimpleDetector(self.lang_script) if lang_script is not None else None
        self.advanced_detector = AdvancedDetector(self.lang_script, valid_data_mgr=valid_data_mgr, conf=Conf()) if HAS_LIB_TORCH and lang_script is not None else None

    def detect_simple(self, words: Sequence[str]) -> Optional[str]:
        if not self.simple_detector:
            return None
        scripts = set(sp(''.join(words))[-1]['details'].keys())
        if lang := self.simple_detector.detect_by_script(scripts):
            return lang
        chars = reduce(op.or_, map(set, words))
        if lang := self.simple_detector.detect_by_chars(chars):
            return lang
        return None

    def detect_advanced(self, _) -> Optional[str]:
        raise NotImplementedError
