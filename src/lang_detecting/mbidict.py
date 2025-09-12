from __future__ import annotations

from typing import Hashable, Iterable


def mbidict( **kwargs) -> MBidict:
    return MBidict(**kwargs)


class KeySet(set):
    def __init__(self, mbidict: MBidict, key: Hashable):
        self._mbididct: MBidict = mbidict
        self._key = key
        super().__init__()

    def add(self, elem: Hashable) -> None:
        if elem not in self:
            super().add(elem)
            self._mbididct[elem].add(self._key)

    def remove(self, elem: Hashable) -> None:
        if elem in self:
            super().remove(elem)
            self._mbididct[elem].remove(self._key)

    def __del__(self):
        for elem in self:
            self._mbididct[elem].remove(self._key)

class MBidict(dict):  # TODO: Think of changing the upper class to Box
    def __init__(self, **kwargs):
        super().__init__()
        self.update(**kwargs)

    def __setitem__(self, left: Hashable, right: Hashable | Iterable) -> None:
        if isinstance(right, Iterable) and not isinstance(right, Hashable):
            for r in right:
                self.__setitem__(left, r)
            return

        super().setdefault(left, KeySet(self, left)).add(right)

    def __getitem__(self, item: Hashable) -> KeySet:
        return super().__getitem__(item)

