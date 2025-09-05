
from __future__ import annotations

import json
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pydash as _
from box import Box
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ConfigMessages:
    LANGUAGE_IN_SAVED: str = 'Language {} is already saved'
    LANGUAGE_NOT_IN_SAVED: str = 'Language {} has not been in saved'


class FileManager:
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
            return Box(yaml.safe_load(f), default_box=True)

    @classmethod
    def load_toml(cls, path: str | Path) -> Box:
        import toml
        with open(path, 'r') as f:
            return Box(toml.load(f),  default_box=True)

    @classmethod
    def save_yaml(cls, path: str | Path, conf: dict) -> None:
        import yaml
        with open(path, 'w+') as f:
            yaml.safe_dump(dict(conf), f, default_flow_style=None, allow_unicode=True, encoding='utf-8')

    @classmethod
    def save_toml(cls, path: str | Path, conf: dict) -> None:
        with open(path, 'w+') as f:
            raise NotImplementedError



class ConfSchema(BaseModel):
    assume: str
    colour: str
    groupby: str
    indirect: str
    langs: list[str]
    mappings: dict[str, dict[str, str] | list[dict[str, str]]]
    member_sep: str = Field(..., alias='member-sep')

    # @field_validator('flag', mode='before')
    # @classmethod
    # def validate_bool_strings(cls, v): ...

class ConfManager:
    def __init__(self, conf_file: Path | str):
        self._conf_file: Path = Path(conf_file)
        self._conf: Optional[Box] = None

    @property
    def conf(self) -> Box:
        return self._conf or self.load_conf()

    def load_conf(self) -> Box:
        self._conf = FileManager.load(self._conf_file); logging.debug(f'Default Config: {json.dumps(self._conf, indent=4, ensure_ascii=False)}')
        return self._conf

    def update_conf(self, parsed: Namespace) -> None:
        logging.debug('Updating Conf')
        if parsed.set:
            raise NotImplementedError('Setting values is not yet supported')
        self._update_add_conf(parsed.add)  # TODO: bad pattern, modifying inside
        self._update_del_conf(parsed.delete)

        # cls.update_lang_order(context.all_langs, conf)
        FileManager.save_yaml(self._conf_file, self.conf.to_dict())

    def _update_add_conf(self, add_bundles: list[list[str]]) -> None:
        for add_bundle in add_bundles:
            key, *vals = add_bundle
            # TODO: Replace with schema  # TODO: make both lang(s) work
            if key.startswith('lang'):
                self.conf.langs.extend(vals)
            else:
                raise NotImplementedError('Only lang-adding is currently supported')

    def _update_del_conf(self, del_bundles: list[list[str]]) -> None:
        for add_bundle in del_bundles:
            key, *vals = add_bundle
            # TODO: Replace with schema  # TODO: make both lang(s) work
            if key.startswith('lang'):
                for val in vals:
                    if val in self.conf.langs:
                        self.conf.langs.remove(val)
            else:
                raise NotImplementedError('Only lang-removing is currently supported')

    def update_lang_order(self, used_langs: list[str]) -> None:
        logging.debug('Updating lang order')
        saved_used = _.filter_(used_langs, self.conf.langs.__contains__)
        saved_unused = _.reject(self.conf.langs, saved_used.__contains__)
        newly_ordered_saved = saved_used + saved_unused
        logging.debug(f'Saved used: {saved_used}\nOld Order: {self.conf.langs}\nNew Order: {newly_ordered_saved}')
        self.conf.langs = newly_ordered_saved
        FileManager.save_yaml(self._conf_file, self.conf.to_dict())
