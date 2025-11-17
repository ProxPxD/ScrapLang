from functools import reduce
from pathlib import Path
from typing import Sequence, Optional

from GlotScript import sp
from pandas import DataFrame
from sympy.core.cache import cached_property

from src.constants import Paths
from src.lang_detecting.preprocessing.data import DataPreprocessor
from src.lang_detecting.lang_predictor import SimpleDetector
from src.resouce_managing.file import FileMgr
from src.lang_detecting.preprocessing.data import Columns as C
import ast
import operator as op

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

print(f'Torch is {("un", "")[has_torch]}available')
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))



class Detector:
    def __init__(self, valid_data_file: Path | str, lang_to_script_file: Path | str):
        self._valid_data_mgr = FileMgr(valid_data_file)
        self._lang_to_script_mgr = FileMgr(lang_to_script_file)
        self.data_preprocessor = DataPreprocessor()
        self.simple_detector = SimpleDetector(self.lang_script)

    @cached_property
    def lang_script(self) -> DataFrame:
        try:
            lang_script = self._lang_to_script_mgr.load()
        except FileNotFoundError:
            lang_script = self.reanalyze()
        lang_script[C.SCRIPTS] = lang_script[C.SCRIPTS].apply(ast.literal_eval)
        lang_script[C.CHARS] = lang_script[C.CHARS].apply(set)
        return lang_script

    def reanalyze(self) -> DataFrame:
        valid_data = self._valid_data_mgr.load()
        lang_to_script = self.data_preprocessor.process(valid_data)
        self._lang_to_script_mgr.save(lang_to_script)
        return lang_to_script

    def detect_simple(self, words: Sequence[str]) -> Optional[str]:
        scripts = set(sp(''.join(words))[-1]['details'].keys())
        if lang := self.simple_detector.detect_by_script(scripts):
            return lang
        chars = reduce(op.or_, map(set, words))
        if lang := self.simple_detector.detect_by_chars(chars):
            return lang
        return None


detector = Detector(Paths.VALID_DATA_FILE, Paths.LANG_SCRIPT_FILE)
detector.reanalyze()
pred = detector.detect_simple('mieć')
print(pred)

# lang_script = sp.create_lang_script_correspondence()
# script_predictor = LangPredictor(lang_script)
# words = [
#     ['spać'],
#     ['мати'],
#     ['食べる'],
#     ['食'],
# ]
# for group in words:
#     pred = script_predictor.predict_lang(group)
#     print(pred)
#
#
# sp.create_script_set_model_groups()
