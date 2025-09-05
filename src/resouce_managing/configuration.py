
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import pydash as _
from box import Box
from pydantic import BaseModel, Field

from .file import FileManager


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
        self._file_manager = FileManager(conf_file)

    @property
    def conf(self) -> Box:
        return self._file_manager.content

    def update_conf(self, parsed: Namespace) -> None:
        logging.debug('Updating Conf')
        if parsed.set:
            raise NotImplementedError('Setting values is not yet supported')
        self._update_add_conf(parsed.add)  # TODO: bad pattern, modifying inside
        self._update_del_conf(parsed.delete)
        self._file_manager.save()

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
        self._file_manager.save()