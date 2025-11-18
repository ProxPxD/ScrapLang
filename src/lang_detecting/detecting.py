import operator as op
from functools import reduce
from typing import Sequence, Optional

from GlotScript import sp
from pandas import DataFrame

from src.lang_detecting.lang_predictor import SimpleDetector

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
#
# print(f'Torch is {("un", "")[has_torch]}available')
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA device count:", torch.cuda.device_count())
#     print("Current device:", torch.cuda.current_device())
#     print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))



class Detector:
    def __init__(self, lang_script: DataFrame):
        self.lang_script = lang_script
        self.simple_detector = SimpleDetector(self.lang_script)


    def detect_simple(self, words: Sequence[str]) -> Optional[str]:
        scripts = set(sp(''.join(words))[-1]['details'].keys())
        if lang := self.simple_detector.detect_by_script(scripts):
            return lang
        chars = reduce(op.or_, map(set, words))
        if lang := self.simple_detector.detect_by_chars(chars):
            return lang
        return None
