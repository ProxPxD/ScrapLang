import json
import logging
from pathlib import Path

from box import Box


class FileManager:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._content = None

    @property
    def content(self):
        return self._content or self.load()

    def refresh(self) -> None:
        self._content = None

    def load(self):
        self._content = self.load_file(self.path)
        return self._content

    def save(self, content = None) -> None:
        self.save_file(self.path, content or self.content)

    @classmethod
    def _get_file_extension(cls, path: str | Path) -> str:
        match Path(path).suffix:
            case '.yaml' | '.yml': return 'yaml'
            case '.toml': return 'toml'
            case _: raise ValueError(f'Unsupported file format: {path}')

    @classmethod
    def _to_dict(cls, content) -> dict:
        match content:
            case Box(): return content.to_dict()
            case _: return dict(content)

    @classmethod
    def load_file(cls, path: str | Path = None) -> Box:
        ext = cls._get_file_extension(path)
        load = getattr(cls, f'load_{ext}')
        content = load(path); logging.debug(f'Loaded file "{path}": {json.dumps(content, indent=4, ensure_ascii=False)}')
        return content

    @classmethod
    def load_yaml(cls, path: str | Path) -> Box:
        import yaml
        with open(path, 'r') as f:
            return Box(yaml.safe_load(f), default_box=True)

    @classmethod
    def load_toml(cls, path: str | Path) -> Box:
        import toml
        with open(path, 'r') as f:
            return Box(toml.load(f),  default_box=True)

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

