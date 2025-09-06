from pathlib import Path

from box import Box
from pydantic import BaseModel, Field

from src.resouce_managing.file import FileMgr


class ShortMemSchema(BaseModel):
    translation: dict = Field(default_factory=dict)
    inflection: dict = Field(default_factory=dict)
    definition: dict = Field(default_factory=dict)

class ShortMemMgr:
    def __init__(self, conf_file: Path | str):
        self._file_mgr = FileMgr(conf_file)

    @property
    def mem(self) -> Box:
        return self._file_mgr.load()


