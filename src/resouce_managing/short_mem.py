from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional

from box import Box
from pydantic import BaseModel, Field

from .file import FileMgr


class MemRecord(BaseModel):
    langs: list
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def time(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)

class ShortMem(BaseModel):
    translation: list[MemRecord] = []
    inflection: list[MemRecord] = []
    definition: list[MemRecord] = []


class ShortMemMgr:
    def __init__(self, conf_file: Path | str, length: int = 8):
        self._file_mgr = FileMgr(conf_file)
        self._length = length
        self._mem: Optional[ShortMem] = None

    @property
    def mem(self) -> ShortMem:
        self._mem = self._mem or ShortMem(**(self._file_mgr.load() or ShortMem().model_dump()))
        return self._mem

    def add(self, parsed: Namespace) -> None:
        if parsed.test:
            return
        if parsed.to_langs:
            self.mem.translation.append(MemRecord(langs=[parsed.from_lang] + parsed.to_langs))
        if parsed.inflection:
            self.mem.inflection.append(MemRecord(langs=[parsed.from_lang]))
        if parsed.definition:
            self.mem.defnition.append(MemRecord(langs=[parsed.from_lang]))
        self._trim_records()
        self._file_mgr.save(self.mem.model_dump())

    def _trim_records(self) -> None:
        self._filter_by_date()
        self._trim_lengths()

    def _filter_by_date(self) -> None:
        ...  # TODO: anhi implement

    def _trim_lengths(self) -> None:
        for key in ShortMem().model_dump().keys():
            records = getattr(self.mem, key)
            setattr(self.mem, key, records[-self._length:])