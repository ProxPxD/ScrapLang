import pydash as _
from pydash import chain as c

from src.lang_detecting.mbidict import MBidict


class LangNarrower:
    def __init__(self, lang_script: MBidict):
        self.lang_script = lang_script

    def narrow_langs(self, scripts: set[str]) -> set[str]:
        pot_langs = set(c(scripts).map(self.lang_script.scripts.get).map(list).flatten().value())
        narroweds = _.filter_(pot_langs, lambda lang: not (scripts - self.lang_script.langs[lang]))
        return set(narroweds)
