from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, Any, Callable

import pandas as pd
import pydash as _
from box import Box
from pandas import DataFrame
from pandas.errors import EmptyDataError
from pydantic import BaseModel, RootModel

UNSET = object()


class FileMgr:
    def __init__(self, path: str | Path, *, func: Callable[[Any], Any] | type = UNSET, create_if_not: bool = False):
        self.path = Path(path)
        if create_if_not:
            self.path.touch(exist_ok=True)
        self._func = func if func is not UNSET else _.identity
        self._content = None

    @property
    def content(self):
        return self._content if self._content is not None else self.load()

    def refresh(self) -> FileMgr:
        self._content = None
        return self

    def load(self, func: Callable[[Any], Any] | type = UNSET):
        self._content = self.load_file(self.path, func=func if func is not UNSET else self._func)
        return self._content

    def is_loaded(self) -> bool:
        return bool(self._content)

    def save(self, content = None) -> FileMgr:
        self.save_file(self.path, content if content is not None else self.content)
        return self.refresh()

    @classmethod
    def _get_file_extension(cls, path: str | Path) -> str:
        match Path(path).suffix:
            case '.yaml' | '.yml': return 'yaml'
            case '.toml': return 'toml'
            case '.json' | '.jsonl': return 'json'
            case '.csv': return 'csv'
            case '.txt': return 'txt'
            case _: raise ValueError(f'Unsupported file format: {path}')

    @classmethod
    def _to_dict(cls, content) -> dict:
        match content:
            case BaseModel() | RootModel(): return content.model_dump()
            case Box(): return content.to_dict()
            case _: return dict(content)

    @classmethod
    def load_file(cls, path: str | Path, func: Callable[[Any], Any] | type = None) -> Box | dict | Any:
        ext = cls._get_file_extension(path)
        load = getattr(cls, f'load_{ext}')
        content = load(path)
        content_view = json.dumps(content, indent=4, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        logging.debug(f'Loaded file "{path}": {content_view}')
        if func:
            content = func(content)
        return content

    @classmethod
    def load_yaml(cls, path: str | Path) -> dict | list | str:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @classmethod
    def load_toml(cls, path: str | Path) -> dict | list | str:
        import toml
        with open(path, 'r') as f:
            return toml.load(f)

    @classmethod
    def load_json(cls, path: str | Path) -> dict | list | str:
        import json
        with open(path, 'r') as f:
            return json.load(f)

    @classmethod
    def load_csv(cls, path: str | Path) -> Optional[DataFrame]:
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return None

    @classmethod
    def load_txt(cls, path: str | Path) -> str:
        with open(path, 'r') as f:
            return '\n'.join(f.readlines())

    @classmethod
    def save_file(cls, path: str | Path = None, content = None) -> None:
        ext = cls._get_file_extension(path)
        save = getattr(cls, f'save_{ext}')
        save(path, content)
        content_view = json.dumps(content, indent=4, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        logging.debug(f'Saved file "{path}": {content_view}')

    @classmethod
    def save_yaml(cls, path: str | Path, conf: dict) -> None:
        import yaml
        with open(path, 'w+') as f:
            yaml.safe_dump(cls._to_dict(conf), f, default_flow_style=None, allow_unicode=True, encoding='utf-8', width=120)

    @classmethod
    def save_toml(cls, path: str | Path, conf: dict) -> None:
        with open(path, 'w+') as f:
            raise NotImplementedError

    @classmethod
    def save_csv(cls, path: str | Path, data: DataFrame) -> None:
        data.to_csv(path, index=False)

    @classmethod
    def save_txt(cls, path: str | Path, text: str) -> None:
        with open(path, 'w') as f:
            f.write(text)
