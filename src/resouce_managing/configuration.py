from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import pydash as _

from .file import FileMgr
from .valid_data import ValidDataMgr
from .. import context_domain
from ..conf_domain import Conf


class ConfMgr:
    def __init__(self, conf_file: Path | str, valid_data_mgr: ValidDataMgr = None):
        self.valid_data_mgr = valid_data_mgr
        self._file_mgr = FileMgr(conf_file, func=lambda conf: Conf(**(conf or {})))

    @property
    def conf(self) -> Conf:
        return self._file_mgr.content

    def update_conf(self, parsed: Namespace) -> None:
        logging.debug('Updating Conf')
        if parsed.set:
            raise NotImplementedError('Setting values is not yet supported')
        self._update_add_conf(parsed.add)  # TODO: bad pattern, modifying inside
        self._update_del_conf(parsed.delete)
        self._file_mgr.save()

    def _update_add_conf(self, add_bundles: list[list[str]]) -> None:
        for add_bundle in add_bundles:
            key, *vals = add_bundle
            # TODO: Replace with schema  # TODO: make both lang(s) work
            if key.startswith('lang'):
                self.conf.langs.extend(vals)
            else:
                raise NotImplementedError('Only lang-adding is currently supported')

    def _update_del_conf(self, del_bundles: list[list[str]]) -> None:
        for del_bundle in del_bundles:
            key, *vals = del_bundle
            # TODO: Replace with schema  # TODO: make both lang(s) work
            if key.startswith('lang'):
                for val in vals:
                    if val in self.conf.langs:
                        self.conf.langs.remove(val)
                        self.valid_data_mgr.remove_entries_of_lang(val)
            else:
                raise NotImplementedError('Only lang-removing is currently supported')

    def update_lang_order(self, used_langs: list[str]) -> None:
        if self.conf.langs is context_domain.UNSET:
            logging.debug('No langs set, not updating langs')
            return
        logging.debug('Updating lang order')
        saved_used = _.filter_(used_langs, self.conf.langs.__contains__)
        saved_unused = _.reject(self.conf.langs, saved_used.__contains__)
        newly_ordered_saved = saved_used + saved_unused
        logging.debug(f'Saved used: {saved_used}\nOld Order: {self.conf.langs}\nNew Order: {newly_ordered_saved}')
        self.conf.langs = newly_ordered_saved
        self._file_mgr.save(self.conf.model_dump(exclude_unset=True))
