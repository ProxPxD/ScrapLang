import json
import logging
from pathlib import Path

from box import Box


class FileMgr:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._content = None

    @property
    def content(self):
        return self._content or self.load()

    def refresh(self) -> None:
        self._content = None

    def load(self, as_box=True):
        self._content = self.load_file(self.path, as_box=as_box)
        return self._content

    def save(self, content = None) -> None:
        self.save_file(self.path, content or self.content)

    @classmethod
    def _get_file_extension(cls, path: str | Path) -> str:
        match Path(path).suffix:
            case '.yaml' | '.yml': return 'yaml'
            case '.toml': return 'toml'
            case '.json' | '.jsonl': return 'json'
            case _: raise ValueError(f'Unsupported file format: {path}')

    @classmethod
    def _to_dict(cls, content) -> dict:
        match content:
            case Box(): return content.to_dict()
            case _: return dict(content)

    @classmethod
    def load_file(cls, path: str | Path = None, as_box=True) -> Box:
        ext = cls._get_file_extension(path)
        load = getattr(cls, f'load_{ext}')
        content = load(path); logging.debug(f'Loaded file "{path}": {json.dumps(content, indent=4, ensure_ascii=False)}')
        if as_box:
            content = Box(content or {}, default_box=True)
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
    def save_file(cls, path: str | Path = None, content = None) -> None:
        ext = cls._get_file_extension(path)
        save = getattr(cls, f'save_{ext}')
        content = save(path, content); logging.debug(f'Saved file "{path}": {json.dumps(content, indent=4, ensure_ascii=False)}')
        return content

    @classmethod
    def save_yaml(cls, path: str | Path, conf: dict | Box) -> None:
        import yaml
        with open(path, 'w+') as f:
            yaml.safe_dump(cls._to_dict(conf), f, default_flow_style=None, allow_unicode=True, encoding='utf-8')

    @classmethod
    def save_toml(cls, path: str | Path, conf: dict | Box) -> None:
        with open(path, 'w+') as f:
            raise NotImplementedError

