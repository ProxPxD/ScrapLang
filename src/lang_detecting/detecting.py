import operator as op
from functools import reduce
from typing import Sequence, Optional

from GlotScript import sp
from pandas import DataFrame

from src.lang_detecting.advanced_predictor import AdvancedDetector
from src.lang_detecting.simple_detector import SimpleDetector

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


class Detector:
    def __init__(self, lang_script: DataFrame):
        self.lang_script = lang_script
        self.simple_detector = SimpleDetector(self.lang_script)
        self.advanced_detector = AdvancedDetector() if has_torch else None

    def detect_simple(self, words: Sequence[str]) -> Optional[str]:
        scripts = set(sp(''.join(words))[-1]['details'].keys())
        if lang := self.simple_detector.detect_by_script(scripts):
            return lang
        chars = reduce(op.or_, map(set, words))
        if lang := self.simple_detector.detect_by_chars(chars):
            return lang
        return None

    def detect_advanced(self, _) -> Optional[str]:
        raise NotImplementedError
