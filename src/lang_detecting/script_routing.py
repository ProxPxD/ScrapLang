from src.lang_detecting.mbidict import MBidict


class ScriptRouter:
    def __init__(self, lang_script: MBidict):
        self.lang_script = lang_script