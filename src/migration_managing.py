from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from functools import cached_property

from packaging.version import Version

from src.constants import Paths
from src.resouce_managing.file import FileMgr
from src.resouce_managing.valid_data import VDC, ValidDataMgr

migrations: dict[Version, list[Callable]] = {}

def version(vers: str):
    def deco(func):
        migrations.setdefault(Version(vers), []).append(func)
        return func
    return deco

class MigrationManager:
    def __init__(self, valid_data_mgr: ValidDataMgr = None):
        self.curr_version = Version('3.8.1')
        self.version_file_mgr = FileMgr(Paths.VERSION_FILE, create_if_not=True)
        self.last_version = Version(self.version_file_mgr.load() or '3.7.1')

        self.valid_data_file = valid_data_mgr.valid_data_file_mgr

    @cached_property
    def needed_migrations(self) -> Collection[Callable]:
        global migrations
        return [migration for vers, migrations in migrations.items() for migration in migrations if vers > self.last_version]

    def is_migration_needed(self) -> bool:
        return bool(self.needed_migrations)

    def migrate(self) -> None:
        for migration in self.needed_migrations:
            logging.debug(f'Running migration "{migration.__name__}"')
            migration(self)
            logging.debug(f'migration "{migration.__name__}" successfully run')
        self.version_file_mgr.save(str(self.curr_version))

    @version('3.8.1')
    def add_is_mapped_field_to_valid_data(self):
        vd = self.valid_data_file.load()
        vd.insert(2, VDC.IS_MAPPED, False)
        self.valid_data_file.save(vd).refresh()
