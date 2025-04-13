
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pydash as _
from box import Box

from .constants import Paths
from .context import Context


@dataclass(frozen=True)
class ConfigMessages:
    LANGUAGE_IN_SAVED: str = 'Language {} is already saved'
    LANGUAGE_NOT_IN_SAVED: str = 'Language {} has not been in saved'


class ConfHandler:
    @classmethod
    def load(cls, path: str | Path) -> Box:
        match Path(path).suffix:
            case '.yaml' | '.yml': return cls.load_yaml(path)
            case '.toml': return cls.load_toml(path)
            case _: raise ValueError(f'Unsupported file format: {path}')

    @classmethod
    def load_yaml(cls, path: str | Path) -> Box:
        import yaml
        with open(path, 'r') as f:
            return Box(yaml.safe_load(f))

    @classmethod
    def load_toml(cls, path: str | Path) -> Box:
        import toml
        with open(path, 'r') as f:
            return Box(toml.load(f))

    @classmethod
    def save_yaml(cls, path: str | Path, conf: dict) -> None:
        import yaml
        with open(path, 'w+') as f:
            yaml.safe_dump(dict(conf), f, default_flow_style=None)

    @classmethod
    def save_toml(cls, path: str | Path, conf: dict) -> None:
        with open(path, 'w+') as f:
            raise NotImplementedError


class ConfUpdater:
    @classmethod
    def update_conf(cls, context: Context) -> None:
        logging.debug('Updating Conf')
        conf: Box = ConfHandler.load(Paths.CONF_FILE)
        cls.update_langs(context, conf)
        ConfHandler.save_yaml(Paths.CONF_FILE, conf.to_dict())

    @classmethod
    def update_langs(cls, context: Context, conf: Box) -> None:
        saved_used = _.filter_(context.all_langs, conf.langs.__contains__)
        saved_unused = _.reject(conf.langs, saved_used.__contains__)
        newly_ordered_saved = saved_used + saved_unused
        logging.debug(f'Saved used: {saved_used}\nOld Order: {conf.langs}\nNew Order: {newly_ordered_saved}')
        conf.langs = newly_ordered_saved
