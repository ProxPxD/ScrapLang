
from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pydash as _
from box import Box

from .constants import Paths
from .context import Context
from pydantic import BaseModel, Field, validator, field_validator


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

class ConfUpdater:
    
    @classmethod
    def load_conf(cls) -> Box:
        return ConfHandler.load(Paths.CONF_FILE)
    
    @classmethod
    def update_conf(cls, parsed: Namespace) -> None:
        logging.debug('Updating Conf')
        conf: Box = cls.load_conf()
        if parsed.set:
            raise NotImplementedError('Setting values is not yet supported')
        cls._update_add_conf(parsed.add, conf)  # TODO: bad pattern, modifying inside
        cls._update_del_conf(parsed.delete, conf)

        # cls.update_lang_order(context.all_langs, conf)
        ConfHandler.save_yaml(Paths.CONF_FILE, conf.to_dict())

    @classmethod
    def _update_add_conf(cls, add_bundles: list[list[str]], conf: Box) -> None:
        for add_bundle in add_bundles:
            key, *vals = add_bundle
            # TODO: Replace with schema  # TODO: make both lang(s) work
            if key.startswith('lang'):
                conf.langs.extend(vals)
            else:
                raise NotImplementedError('Only lang-adding is currently supported')

    @classmethod
    def _update_del_conf(cls, del_bundles: list[list[str]], conf: Box) -> None:
        for add_bundle in del_bundles:
            key, *vals = add_bundle
            # TODO: Replace with schema  # TODO: make both lang(s) work
            if key.startswith('lang'):
                for val in vals:
                    if val in conf.langs:
                        conf.langs.remove(val)
            else:
                raise NotImplementedError('Only lang-removing is currently supported')

    @classmethod
    def update_lang_order(cls, used_langs: list[str], conf: Box = None) -> None:
        logging.debug('Updating lang order')
        conf = conf or cls.load_conf()
        saved_used = _.filter_(used_langs, conf.langs.__contains__)
        saved_unused = _.reject(conf.langs, saved_used.__contains__)
        newly_ordered_saved = saved_used + saved_unused
        logging.debug(f'Saved used: {saved_used}\nOld Order: {conf.langs}\nNew Order: {newly_ordered_saved}')
        conf.langs = newly_ordered_saved
        ConfHandler.save_yaml(Paths.CONF_FILE, conf.to_dict())
