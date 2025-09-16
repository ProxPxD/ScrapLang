from typing import Sequence, Optional

from src.lang_detecting.lang_narrower import LangNarrower
from src.lang_detecting.mbidict import MBidict
from GlotScript import sp

from pydash import chain as c
import pydash as _

class LangPredictor:
    def __init__(self, lang_script: MBidict):
        self.lang_narrower = LangNarrower(lang_script)

    def predict_lang(self, words: Sequence[str]) -> Optional[str]:
        scripts = set(sp(''.join(words))[-1]['details'].keys())
        langs = self.lang_narrower.narrow_langs(scripts)
        if len(langs) == 1:
            return  next(iter(langs))
