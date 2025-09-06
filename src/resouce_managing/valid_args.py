from pathlib import Path

from pydantic import RootModel, field_validator

from .file import FileMgr
from ..constants import supported_languages


class ValidArgs(RootModel):
    root: dict[str, list[str]]

    @field_validator('root', mode='before')
    def val_keys(self, d: dict):
        supported = set(supported_languages.keys())
        actual = set(d.keys())
        if not (invalid := actual / supported):
            raise ValueError(f'"{list(invalid)}" languages are not supported')
        return d

class ValidArgsMgr:
    def __init__(self, conf_file: Path | str):
        self._file_mgr = FileMgr(conf_file)

